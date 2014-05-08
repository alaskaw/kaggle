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
library(xtable)
library(pbapply)

# Vary: formula, validate (vs x-val.), scorer (ROC vs. Acc.), selector (best vs oneSE)

# Setup
# ----------------------------------------------------------------------------
setwd('~/Code/kaggle/titanic/')
load('data/mungedData.RData')

dualcol = c("brown2", "cornflowerblue") #c("black", "grey80")

doParallel = T
validate = F
trainrows = createDataPartition(train$survived, p=0.8, list=F)
trainsubset = train[trainrows, ]
validationsubset = train[-trainrows, ]

trainset = if(validate) trainsubset else train

if(doParallel){
  library(doMC)
  registerDoMC(cores=6)
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
rseed = 43
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
fmla0 = survived ~ pclass + sex + age + fare + embarked + familysize
fmla1 = survived ~ pclass + sex + age + fare + embarked + familysize + title
fmla2 = survived ~ pclass + sex + age + child + farecat + embarked + title + familysizefac + familyid 
fmla3 = survived ~ pclass + sex + age + I(embarked=='S') + title + I(title=="Mr" & pclass=="3") + familysize 

fmla = fmla1

# ----------------------------------------------------------------------------
# Set up models
# ----------------------------------------------------------------------------
# alpha controls relative weight of lasso and ridge constraints (1=lasso, 0=ridge)
# lambda is the regularization parameter
glmnetgrid = expand.grid(.alpha = seq(0, 1, 0.1), .lambda = seq(0, 1, 0.1))
rfgrid = data.frame(.mtry = c(2,3))
gbmgrid = expand.grid(.interaction.depth = c(1, 5, 9), .n.trees = (1:15)*100, .shrinkage = 0.1)
adagrid = expand.grid(.iter = c(50, 100), .maxdepth = c(4, 8), .nu = c(0.1, 1))
svmgrid = expand.grid(.sigma=c(0.1, 0.25, 0.5, 0.75, 1), .C=c(0.1, 1, 2, 5, 10))
cforgrid = expand.grid(mtry = c(2,3))
cforcontrols = cforest_unbiased(ntree = 2000)
blackgrid = expand.grid(.mstop=c(500), .maxdepth=c(5,10))
earthgrid = expand.grid(.nprune=c(10), .degree=c(1,2))
gambogrid = expand.grid(.mstop=c(500), .prune=c(10))
logitgrid = expand.grid(.nIter=c(10,50,100))

pp = c("center", "scale")

configs = list()
configs$glmnet = list(method="glmnet", tuneGrid=glmnetgrid, preProcess=pp)
configs$rf = list(method="rf", tuneGrid=rfgrid, preProcess=NULL, ntree=2000)
configs$gbm = list(method="gbm", tuneGrid=gbmgrid, preProcess=NULL)
configs$ada = list(method="ada", tuneGrid=adagrid, preProcess=NULL)
configs$svm = list(method="svmRadial", tuneGrid=svmgrid, preProcess=pp)
##configs$cforest = list(method="cforest", tuneGrid=cforgrid, preProcess=NULL)
configs$cforest = list(method="cforest", controls=cforcontrols, preProcess=NULL)
configs$black = list(method="blackboost", tuneGrid=blackgrid, preProcess=pp)
configs$earth = list(method="earth", tuneGrid=earthgrid, preProcess=pp)
configs$gambo = list(method="gamboost", tuneGrid=gambogrid, preProcess=pp)
configs$logit = list(method="LogitBoost", tuneGrid=logitgrid, preProcess=pp)
configs$bayesglm = list(method="bayesglm", preProcess=pp)

# ----------------------------------------------------------------------------
# Train them up
# ----------------------------------------------------------------------------
arg = list(form = fmla, data = trainset, trControl = cvctrl, metric = scorer)
models = list()
set.seed(rseed)
for(i in 1:length(configs)) 
{
  cat(sprintf("Training %s ...\n", configs[[i]]$method)); flush.console();
  models[[i]] = do.call("train.formula", c(arg, configs[[i]]))
}

# Didn't work?
models["LogitBoost"] = NULL

names(models) = sapply(models, function(x) x$method)

rowToTab = function(row, rnames=NULL, transp=T){
  ms = data.frame(row)
  if(transp)
    ms = t(ms)
  rownames(ms) = rnames
  return(print(xtable(ms, digits=3), type="html", include.rownames=T, html.table.attributes=NULL))
}

# All scores
print("Scores max & min:")
sort(sapply(models, function(x) max(x$results[scorer]) ))

# Get training and prediction errors
for(i in 1:length(models)){
  models[[i]]$trainAcc = modelAccuracy(models[[i]], trainset)
}
trainAccs = sapply(models, function(x) x$trainAcc)
rowToTab(trainAccs)

if(validate){
  for(i in 1:length(models)){
    models[[i]]$conMat = confusionMatrix(predict(models[[i]], validationsubset), validationsubset$survived) 
  }
  
  valAccs = sapply(models, function(x) x$conMat$overall['Accuracy'][[1]])
  accs = t(data.frame(train = trainAccs, val = valAccs))
}


# ----------------------------------------------------------------------------
# Compare models visually
# ----------------------------------------------------------------------------
pal = brewer.pal(10,"Paired")
if(validate){
  for(i in 1:length(models)){
    probs = predict(models[[i]], validationsubset, type="prob")  
    if(i==1) plot.roc(validationsubset$survived, probs$yes, percent=T, col=pal[[i]])
    else lines.roc(validationsubset$survived, probs$yes, percent=T, col=pal[[i]])
  }
  legend("bottomright", legend=names(models), col=pal, lwd=2, cex=0.7)
}

# Compare resample performances
resamps = resamples(models)
summary(resamps)
trellis.par.set(caretTheme())
bwplot(resamps, layout=c(3,1))
dotplot(resamps, layout=c(3,1))
xyplot(resamps) 

diffs = diff(resamps)
diffs
summary(diffs)
dotplot(diffs)


# ----------------------------------------------------------------------------
# Create ensembles
# ----------------------------------------------------------------------------
# Greedy ensemble
greedyEns = caretEnsemble(models, iter=1000L)
greedyW = sort(greedyEns$weights, decreasing=T)
greedyEns$error

# Linear regression ensemble
linearEns = caretStack(models, method='glm', trControl = cvctrl, metric = scorer)
linearEns$error


# Compare models to ensembles
aucs = NULL
if(validate){
  preds = data.frame(sapply(models, function(x) predict(x, validationsubset, type='prob')[,2]))
  preds$greedyEns = predict(greedyEns, validationsubset)
  preds$linearEns = predict(linearEns, validationsubset, type='prob')[,2]
  aucs = sort(data.frame(colAUC(preds, validationsubset$survived)))
  rowToTab(aucs)
}
else{
  muroc = summary(resamps)$statistics$ROC[,"Mean"]
  musens = summary(resamps)$statistics$Sens[,"Mean"]
  muspec = summary(resamps)$statistics$Spec[,"Mean"]
  aucs = data.frame(ROC=muroc, Sens=musens, Spec=muspec)
  
  aucs["linearEns",] = linearEns$error[c("ROC", "Sens", "Spec")]
  aucs["greedyEns", "ROC"] = greedyEns$error[[1]]
  print(xtable(t(aucs), digits=3), type="html")
}

# Submission
# Test set has same columns as training set but misses the target variable ($survived)
# ----------------------------------------------------------------------------
predProbL = predict(linearEns, test, type='prob')
predL = ifelse(predProbL$no > 0.5, 1, 0)
predProbG = predict(greedyEns, test)
predG = ifelse(predProbG > 0.5, 1, 0)
predProbB = predict(models$gbm, test, type='prob')
predB = ifelse(predProbB$yes > 0.5, 1, 0)

#levels(pred) = c(0,1)

submit = data.frame(PassengerId=test$passengerid, Survived=predL)
write.csv(submit, file="data/prediction.csv", row.names=F)
