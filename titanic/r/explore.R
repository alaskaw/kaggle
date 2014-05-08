# ----------------------------------------------------------------------------
# Predict survival on the Titanic
# http://www.kaggle.com/c/titanic-gettingStarted/
# ----------------------------------------------------------------------------
rm(list=ls(all=TRUE))

library(Hmisc)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)
library(partykit)
library(vcd)
library(lattice)
library(corrgram)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(scales)
library(caret)
library(e1071)
library(pROC)
library(ada)
library(kernlab)
library(doMC)
library(gbm)

setwd('~/Code/kaggle/titanic/')
source('../lib/r/ggplotContBars.R')

dualcol = brewer.pal(3, "Set1")
duallight = brewer.pal(3, "Pastel1")
cbbPalette = c("#E69F00", "#56B4E9", "#000000", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
#dualcol = c("black", "grey80")

registerDoMC(cores=7)

errRate = function(pred, lab){
  return(100*sum(pred != lab) / length(pred))
}

accuracy = function(pred, lab){
  return(1 - errRate(pred, lab)/100)
}

# Compare factor levels for train and test set
compSetFacs = function(var, prop=F){  
  trtab = table(train$var)
  tetab = table(test$var)
  if(prop){
    trtab = prop.table(trtab)
    tetab = prop.table(tetab)
  }
  t = rbind(trtab, tetab)
  rownames(t) = c("train", "test")
  t = data.frame(t(t))
  attributes(t)$name = var
  t
}

# Compare numeric vars for train and test set
compSetNums = function(var){
  tr = summary(train[,var])
  te = summary(test[,var])
  t = rbind(tr, te)
  rownames(t) = c(paste("train_", var,sep=""), paste("test_", var, sep=""))
  t
}

# Load
# ----------------------------------------------------------------------------
load('data/mungedData.RData')

# Easy, separate access to factors and numeric predictors
# ----------------------------------------------------------------------------
str(train)
varclasses = sapply(train,class)
varnames = names(train)
facvars = which(varnames %in% c('pclass', 'sex', 'embarked', 'farecat', 'familysizefac', 'familyid', 'title')) 
numvars = which(varnames %in% c('age', 'familysize', 'fare'))
predictors = c(facvars, numvars)

# Explore
# ----------------------------------------------------------------------------
# Compare train and test set to check for imbalances
faccompProp = lapply(varnames[facvars], function(x) compSetFacs(x, prop=T))
faccomp = lapply(varnames[facvars], compSetFacs)
#faccomp = lapply(varnames[facvars], function(x) { data.frame(cbind(table(train[,x]), table(test[, x])))})
pvals = sapply(faccomp, function(x) chisq.test(x)$p.value)
print("Unbalanced factors:")
faccomp[[which(pvals<0.05)]]

numcomp = do.call(rbind, lapply(varnames[numvars], compSetNums))
print(numcomp)

# Overall survival rate
prop.table(table(train$survived))
plPropGroup = F
plPos = "dodge"

# By gender: most women survived, most men died
t = table(train$survived, train$sex)
print(prop.table(t, 2)) # Proportion per gender
chisq.test(t)

# By class: prop. more people survive in 1st class, compared to 3rd 
t = table(train$survived, train$pclass)
print(prop.table(t, 2)) # Proportion per gender
chisq.test(t)

# By embarkment (might have influenced location at time of accident)
chisq.test(table(train$survived, train$embarked))

# By age: identify children
summary(train$age)
hist(train$age, 20)
plot(survived ~ age, train)
gg = ggplot(train, aes(x=survived, y=age, fill=survived)) + 
  geom_boxplot(notch=T) + 
  theme_classic(base_size=12) +
  scale_fill_manual(values=dualcol) +
  guides(fill=F)
print(gg)

# probability peak for young childre
ageDensPl = ggplot(train, aes(x=age, fill=survived)) + geom_density(alpha=.3) + 
  theme_classic() + scale_fill_manual(values=dualcol) + guides(fill=F)
print(ageDensPl)
ggplot(train, aes(x=age, fill=survived)) + geom_histogram(alpha=.3) + theme_classic() + scale_fill_manual(values=dualcol)
#ggplot(train, aes(x=survived, y=age, fill=survived)) + geom_boxplot() + guides(fill=FALSE)

# By child: if you're a child you're more likely to survive, as an adult to die.
ggplotContBars(train, 'child', 'survived', propPerGroup=T, position="facet", colors=dualcol)

 # By fare: the higher the fare the greater the probability of survival
summary(train$fare)
hist(train$fare, breaks=40)
barplot(prop.table(table(train$farecat))) 
fareBoxPl = ggplot(train, aes(x=survived, y=fare, fill=survived)) + 
  geom_boxplot(notch=T) + 
  scale_y_log10() +
  theme_classic() + scale_fill_manual(values=dualcol) + guides(fill=F)
print(fareBoxPl)

fareVioPl = ggplot(train, aes(x=survived, y=fare, fill=survived)) + 
  geom_violin(scale="count", trim=F) + 
  scale_y_log10() +
  theme_classic() + scale_fill_manual(values=dualcol) + guides(fill=F)
print(fareVioPl)

# By fare and age
#pairs(train[,c('age','fare')], col=train$survived)
ggplot(train, aes(x=age, y=fare, color=survived)) + 
  geom_point() + 
  geom_smooth(method="lm") +
  theme_classic() + 
  scale_color_manual(values=dualcol)

# Title
str(train$title)

# Family size
str(train$familysize)
hist(train$familysize)
plot(survived ~ familysize, train)  
summary(train$familyid)
famDensPl = ggplot(train, aes(x=familysize, fill=survived)) + geom_density(alpha=.3) + 
  scale_x_log10() +
  theme_classic() + scale_fill_manual(values=dualcol)
print(famDensPl)
famBoxPl = ggplot(train, aes(x=survived, y=familysize, fill=survived)) + geom_boxplot(notch=T) + 
#  scale_y_log10() +
  theme_classic() + scale_fill_manual(values=dualcol)
print(famBoxPl)
ggplot(train, aes(familysize, survived, color=survived)) + 
  geom_point(position=position_jitter(width=1,height=.5))

grid.arrange(fareBoxPl, ageDensPl, nrow=1)

source('../lib/r/ggplotContBars.R')
plvars = c("sex", "pclass", "embarked", "familysizefac", "farecat")
pltitles = c("sex", "class", "embarked", "family size", "fare")
plvars2 = c("child", "title")
pltitles2 = c("is child", "title")

contBarsGrid = function(vars, titles){
  plts = list()
  for (i in 1:length(vars))
    plts[[i]] = ggplotContBars(train, vars[[i]], 'survived', propPerGroup=T, position="dodge", colors=dualcol, title=titles[[i]])
  
  fig = do.call(grid.arrange, c(plts, nrow=1))  
  fig
}

contBarsGrid(plvars, pltitles)
ggplotContBars(train, 'child', 'survived', propPerGroup=T, position="dodge", colors=dualcol, title='is child')
ggplotContBars(train, 'title', 'survived', propPerGroup=T, position="dodge", colors=dualcol, title='title')

# Cabin
str(train$cabcat)
table(train$survived, train$cabcat, useNA="always")
table(train$survived, train$cabeven, useNA="always")
plot(survived~cabcat, train)
plot(survived~cabeven, train)
ggplotContBars(train, 'cabcat', 'survived', propPerGroup=T, position="facet", colors=dualcol)
ggplotContBars(train, 'cabeven', 'survived', propPerGroup=F, position="facet", colors=dualcol)

# Even females in 3rd class who have spend more on ticket die more often (reason not clear)
farenum = factor(train$farecat, labels=seq(length(levels(train$farecat))))
aggregate(survived ~ farenum + pclass + sex, data=train, FUN=function(x) {sum(as.numeric(x))/length(x)})

# Train a decision tree
# ----------------------------------------------------------------------------
plottree = function(tree){
  rpart.plot(tree, type=4, extra=104, branch=0, varlen=0, faclen=0, clip.right.labs=F, tweak=1.0, compress=F, ycompress=F, 
           box.col=c(duallight[1], duallight[2])[tree$frame$yval], fallen.leaves=F, branch.lty=1)
}

dc1 = rpart(survived ~ pclass + sex + age + familysize + fare + embarked, data=train, method="class")
printcp(dc1)
# Print cross validation results as function of complexity parameter
# Used to choose model with greatest complexity whose error is greater than the minimum + 1sd
plotcp(dc1)
summary(dc1)
#fancyRpartPlot(dc1)
plottree(dc1)
imp = sort(dc1$variable.importance / sum(dc1$variable.importance))
dotplot(imp, xlab="importance")


# Get complexity parameter for minimum cross-validated error
# Then prune to that complexity
#cp = dc1$cptable[which.min(dc1$cptable[,"xerror"]),"CP"]
#dc1p = prune(dc1, cp=0.014)
#fancyRpartPlot(dc1p)

dc1pred = predict(dc1, train, type="class")
confusionMatrix(dc1pred, train$survived)

dc2 = rpart(survived ~ pclass + sex + age + familysize + familyid + title, data=train, method="class")
plottree(dc2)
dc2pred = predict(dc2, train, type="class")
confusionMatrix(dc2pred, train$survived)

# Identify wrong predictions: anything that distinguishes them?
wrongPredI = which(dc2pred != train$survived)
wrongPred = train[wrongPredI,]
corrPred = train[-wrongPredI,]

# Histogram of predicted probabilities: well separated
dc2predP = data.frame(predict(dc2, train))
ggplot(dc2predP, aes(x=yes)) + geom_density() + theme_classic()
hist(dc2predP$yes, 10)

dc3 = ctree(survived ~ pclass + sex + age + familysize + familyid + title, data=train)
plot(dc3)
dc3pred = predict(dc3, train, type="response")
confusionMatrix(dc3pred, train$survived)

# Train a logistic regression
# ----------------------------------------------------------------------------
lr1 = glm(survived ~ pclass + sex + age + familysize + fare + embarked, data=trainset, family=binomial("logit"))
lr2 = glm(survived ~ pclass + sex + age + familysize + familysizefac + fare + farecat + embarked + child, data=trainset, family=binomial("logit"))
lr3 = glm(survived ~ pclass + sex + age + familysize + familysizefac + child, data=trainset, family=binomial("logit"))
summary(lr1)
anova(lr1, test="Chisq")
anova(lr2, test="Chisq")
anova(lr3, test="Chisq")

lr1pred = predict.glm(lr1, newdata=testset, type="response")
lr1pred = ifelse(lr1pred > 0.5, 'yes', 'no')
lr2pred = predict.glm(lr2, newdata=testset, type="response")
lr2pred = ifelse(lr2pred > 0.5, 'yes', 'no')
lr3pred = predict.glm(lr3, newdata=testset, type="response")
lr3pred = ifelse(lr3pred > 0.5, 'yes', 'no')

confusionMatrix(lr1pred, testset$survived)
confusionMatrix(lr2pred, testset$survived)
confusionMatrix(lr3pred, testset$survived)
# best accuracy ~0.825


# Train a randomForest
# ----------------------------------------------------------------------------
set.seed(seed)
rf1 = randomForest(survived ~ pclass + sex + age + familysize + fare + embarked + title, data=trainset, importance=TRUE, ntree=500)
varImpPlot(rf1)
rf1pred = predict(rf1, testset)
confusionMatrix(rf1pred, testset$survived)
# accuracy ~ 0.85

# Try a slightly different type of random forest
set.seed(seed)
rf2 = cforest(survived ~ pclass + sex + age + familysize + fare + embarked + title + familyid, data=train, controls=cforest_unbiased(ntree=2000, mtry=3))
rf2
rf2pred = predict(rf2, testset, OOB=T, type="response")
confusionMatrix(rf2pred, testset$survived)
# best accuracy ~0.86,



