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
library(vcd)
library(lattice)
library(corrgram)
library(ggplot2)
library(ggthemes)
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
cbbPalette = c("#E69F00", "#56B4E9", "#000000", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
#dualcol = c("black", "grey80")

registerDoMC(cores=7)

errRate = function(pred, lab){
  return(100*sum(pred != lab) / length(pred))
}

accuracy = function(pred, lab){
  return(1 - errRate(pred, lab)/100)
}

# Load
# ----------------------------------------------------------------------------
load('data/mungedData.RData')
trainrows = createDataPartition(train$survived, p=0.8, list=F)
trainset = train[trainrows, ]
testset = train[-trainrows, ]

# Easy, separate access to factors and numeric predictors
# ----------------------------------------------------------------------------
str(train)
varclasses = sapply(train,class)
varnames = names(train)
facvars = which(varnames %in% c('pclass', 'sex', 'embarked', 'title', 'farecat', 'familysizefac', 'familyid')) 
numvars = which(varnames %in% c('age')) #, 'familysize', 'fare'))
predictors = rbind(facvars, numvars)

# Explore
# ----------------------------------------------------------------------------

# Overall survival rate
prop.table(table(train$survived))

# By gender: most women survived, most men died
t = table(train$survived, train$sex)
print(prop.table(t, 2)) # Proportion per gender
chisq.test(t)
gg = ggplotContBars(train, 'sex', 'survived', propPerGroup=F, position="facet", colors=dualcol)
print(gg)
#ggsave(filename='figures/sex-survival.jpg', plot=gg, width=5, height=5, units="cm")

# By class: prop. more people survive in 1st class, compared to 3rd 
t = table(train$survived, train$pclass)
print(prop.table(t, 2)) # Proportion per gender
chisq.test(t)
ggplotContBars(train, 'pclass', 'survived', propPerGroup=F, position="facet", colors=dualcol)

# By embarkment (might have influenced location at time of accident)
chisq.test(table(train$survived, train$embarked))
ggplotContBars(train, 'embarked', 'survived', propPerGroup=F, position="facet", colors=dualcol)

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
ggplot(train, aes(x=age, fill=survived)) + geom_density(alpha=.3) + theme_classic() + scale_fill_manual(values=dualcol)

# By child: if you're a child you're more likely to survive, as an adult to die.
ggplotContBars(train, 'child', 'survived', propPerGroup=T, position="facet", colors=dualcol)

 # By fare: the higher the fare the greater the probability of survival
summary(train$fare)
hist(train$fare, breaks=40)
barplot(prop.table(table(train$farecat))) 
ggplotContBars(train, 'farecat', 'survived', propPerGroup=F, position="facet", colors=dualcol)

# By fare and age
#pairs(train[,c('age','fare')], col=train$survived)
ggplot(train, aes(x=age, y=fare, color=survived)) + 
  geom_point() + 
  geom_smooth(method="lm") +
  theme_classic() + 
  scale_color_manual(values=dualcol)

# Title
str(train$title)
ggplotContBars(train, 'title', 'survived', propPerGroup=T, position="facet", colors=dualcol)

# Family size
str(train$familysize)
plot(survived ~ familysize, train)  
ggplotContBars(train, 'familysizefac', 'survived', propPerGroup=T, position="facet", colors=dualcol)
summary(train$familyid)

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
dc1 = rpart(survived ~ pclass + sex + age + familysize + fare + embarked, data=trainset, method="class")
printcp(dc1)
plotcp(dc1)
fancyRpartPlot(dc1)

dc1pred = predict(dc1, testset, type='class')
confusionMatrix(dc1pred, testset$survived)

dc2 = rpart(survived ~ pclass + sex + age + familysize + familysizefac + familyid + fare + farecat + embarked + title + child, data=trainset, method="class")
fancyRpartPlot(dc2)
dc2pred = predict(dc2, testset, type="class")
confusionMatrix(dc2pred, testset$survived)

# Identify wrong predictions: anything that distinguishes them?
wrongPredI = which(dc2pred != testset$survived)
wrongPred = train[wrongPredI,]
corrPred = train[-wrongPredI,]

# Histogram of predicted probabilities: well separated
dc2predP = data.frame(predict(dc2, testset))
ggplot(dc2predP, aes(x=yes)) + geom_density() + theme_classic()
hist(dc2predP$yes, 10)

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



