# ----------------------------------------------------------------------------
# Predict survival on the Titanic
# http://www.kaggle.com/c/titanic-gettingStarted/
# ----------------------------------------------------------------------------
rm(list=ls(all=TRUE))

library(Hmisc)
library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)
library(party)
library(vcd)
library(ggplot2)
library(ggthemes)
library(caret)
library(e1071)
library(pROC)
library(ada)
library(kernlab)
library(gbm)
library(glmnet)
library(caretEnsemble)
library(RColorBrewer)

# Setup
# ----------------------------------------------------------------------------
setwd('~/Code/kaggle/titanic/')
load('data/mungedData.RData')

dualcol = c("brown2", "cornflowerblue") #c("black", "grey80")

doParallel = T
validate = T
trainrows = createDataPartition(train$survived, p=0.8, list=F)
trainsubset = train[trainrows, ]
validationsubset = train[-trainrows, ]

trainset = if(validate) trainsubset else train

if(doParallel){
  library(doMC)
  registerDoMC(cores=7)
}

# Helpers
# ----------------------------------------------------------------------------
errRate = function(pred, lab){
  return(sum(pred != lab) / length(pred))
}

myaccuracy = function(pred, lab){
  return(1 - errRate(pred, lab))
}

modelAccuracy = function(model, set){
  return(myaccuracy(predict(model, set), set$survived))
}

# Setup caret package for cross-validated training
# ----------------------------------------------------------------------------
rseed = 42
scorer = 'ROC' # 'ROC' or "Accuracy'
summarizor = if(scorer == 'Accuracy') defaultSummary else twoClassSummary
selector = "best" # "best" or "oneSE"

# Save predictions, data for ensemble creation later on
folds = 10
repeats = 10
cvctrl = trainControl(method="repeatedcv", number=folds, repeats=repeats, p=0.8, 
                      summaryFunction=summarizor, selectionFunction=selector, 
                      classProbs=T, savePredictions=T, returnData=T,
                      index=createMultiFolds(trainset$survived, k=folds, times=repeats))

# Formulas
# ----------------------------------------------------------------------------
# survived ~ pclass + sex + age + child + fare + farecat + embarked + title + familysize + familysizefac + familyid
fmla0 = survived ~ pclass + sex + age
fmla1 = survived ~ pclass + sex + age + fare + embarked + familysizefac + title
fmla2 = survived ~ pclass + sex + age + child + farecat + embarked + title + familysizefac + familyid 
fmla3 = survived ~ pclass + sex + age + I(embarked=='S') + title + I(title=="Mr" & pclass=="3") + familysize 

fmla = fmla1

# ----------------------------------------------------------------------------
# Set up models
# ----------------------------------------------------------------------------
# alpha controls relative weight of lasso and ridge constraints (1=lasso, 0=ridge)
# lambda is the regularization parameter
glmnetgrid = expand.grid(.alpha = seq(0, 1, 0.1), .lambda = seq(0, 1, 0.1))
rfgrid = data.frame(.mtry = 3)
gbmgrid = expand.grid(.interaction.depth = c(1, 5, 9), .n.trees = (1:15)*100, .shrinkage = 0.1)
adagrid = expand.grid(.iter = c(50, 100), .maxdepth = c(4, 8), .nu = c(0.1, 1))
svmgrid = expand.grid(.sigma=c(0.1, 0.25, 0.5, 0.75, 1), .C=c(0.1, 1, 2, 5, 10))
blackgrid = expand.grid(.mstop=c(), .maxdepth=c())
earthgrid = expand.grid(.nprune=c(), .degree=c())
gambogrid = expand.grid(.mstop=c(), .prune=c())
logitgrid = expand.grid(.nIter=c())
bayesgrid = expand.grid(.fL=c(), .usekernel=c())

pp = c("center", "scale")

configs = list()
configs$rf = list(method="rf", tuneGrid=rfgrid, preProcess=NULL, allowParallel=T, ntree=2000)
configs$glmnet = list(method="glmnet", tuneGrid=glmnetgrid, preProcess=pp, allowParallel=T)
configs$gbm = list(method="gbm", tuneGrid=gbmgrid, preProcess=NULL, allowParallel=T)
configs$ada = list(method="ada", tuneGrid=adagrid, preProcess=NULL, allowParallel=T)
configs$svm = list(method="svmRadial", tuneGrid=svmgrid, preProcess=pp, allowParallel=T)
# earth, gam ...

# ----------------------------------------------------------------------------
# Train them up
# ----------------------------------------------------------------------------
arg = list(form = fmla, data = trainset, trControl = cvctrl, metric = scorer)
models = list()
set.seed(rseed)
for(i in 1:length(configs)) 
{
  cat(sprintf("Training %s ...\n", configs[[i]]$method)); flush.console();
  
  # Unfortunately allowParallel is not a train() argument but a trControl() argument
  argms = configs[[i]]
  ctrl = cvctrl
  ctrl$allowParallel = argms$allowParallel
  argms$allowParallel = NULL
  
  models[[i]] = do.call("train.formula", c(arg, argms))
}

names(models) = sapply(models, function(x) x$method)

# All scores
print("Scores max & min:")
sort(sapply(models, function(x) max(x$results[scorer]) ))
sort(sapply(models, function(x) min(x$results[scorer]) ))

# Get training and prediction errors
for(i in 1:length(models)){
  models[[i]]$trainAcc = modelAccuracy(models[[i]], trainset)
}
sort(sapply(models, function(x) x$trainAcc))

if(validate){
  for(i in 1:length(models)){
    models[[i]]$conMat = confusionMatrix(predict(models[[i]], validationsubset), validationsubset$survived) 
  }
}


# ----------------------------------------------------------------------------
# Compare models visually
# ----------------------------------------------------------------------------
pal = brewer.pal(9,"Set1")
if(validate){
  for(i in 1:length(models)){
    probs = predict(models[[i]], validationsubset, type="prob")  
    if(i==1) plot.roc(validationsubset$survived, probs$yes, percent=T, col=pal[[i]])
    else lines.roc(validationsubset$survived, probs$yes, percent=T, col=pal[[i]])
  }
  legend("bottomright", legend=names(models), col=pal, lwd=2)
}

# Compare resample performances
resamps = resamples(models)
summary(resamps)
trellis.par.set(caretTheme())
bwplot(resamps, layout=c(3,1))
dotplot(resamps, metric=scorer)
dotplot(resamps, metric="Sens") # RF wins on sensitivity
dotplot(resamps, metric="Spec") # glmnet wins on specificity

# ----------------------------------------------------------------------------
# Create ensembles
# ----------------------------------------------------------------------------
# Greedy ensemble
greedyEns = caretEnsemble(models, iter=1000L)
sort(greedyEns$weights, decreasing=T)
greedyEns$error

# Linear regression ensemble
linearEns = caretStack(models, method='glm', trControl=trainControl(method='cv'))
linearEns$error

# Compare models to ensembles
if(validate){
  preds = data.frame(sapply(models, function(x) predict(x, validationsubset, type='prob')[,2]))
  preds$greedyEns = predict(greedyEns, validationsubset)
  preds$linearEns = predict(linearEns, validationsubset, type='prob')[,2]
  sort(data.frame(colAUC(preds, validationsubset$survived)))
}

# Submission
# Test set has same columns as training set but misses the target variable ($survived)
# ----------------------------------------------------------------------------
pred = predict(linearEns, test, type='prob')[,2]
pred = predict(greedyEns, test)
pred = predict(models$gbm, test)
levels(pred) = c(0,1)
pred = ifelse(pred > 0.5, 1, 0)

submit = data.frame(PassengerId=test$passengerid, Survived=pred)
write.csv(submit, file="data/prediction.csv", row.names=F)
