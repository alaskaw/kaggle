# ----------------------------------------------------------------------------
# Predict survival on the Titanic
# http://www.kaggle.com/c/titanic-gettingStarted/
# ----------------------------------------------------------------------------

# Import data
rm(list=ls(all=TRUE))
setwd('~/Code/kaggle/titanic/')
train = read.csv("data/train.csv", stringsAsFactors=FALSE)
test = read.csv("data/test.csv", stringsAsFactors=FALSE)

# Create a collated data set for common preprocessing (then split them again later)
ntr = nrow(train)
nte = nrow(test)
test$Survived = NA
data = rbind(train, test)

# Overview of variables
names(data) = tolower(names(data))
names(data)
str(data)

# Factor conversions
data$survived = as.factor(data$survived)
data$pclass = as.factor(data$pclass)
data$sex = as.factor(data$sex)
data$embarked = as.factor(data$embarked)
str(data)

# ----------------------------------------------------------------------------
# Univariate exploration and fixups
# ----------------------------------------------------------------------------

# passsenger id: is unnique identifier and therefore useless
# ----------------------------------------------------------------------------
length(unique(data$passengerid))

# survived: only 38.4% of all passengers survived
# ----------------------------------------------------------------------------
tab = table(train$Survived)
prop.table(tab) 
levels(data$survived) = c("no", "yes")

# pclass: uneven factor levels. there are much more data points in class 3.
# But we assume this isn't a big problem for the algorithms used.
# ----------------------------------------------------------------------------
str(data$pclass)
sum(is.na(data$pclass)) # no NAs
tab = table(data$pclass)
ptab = prop.table(tab)
barplot(ptab)
data$pclass = ordered(data$pclass)

# name: contains formal titles, which can be extracted as a potentially
# useful feature
# ----------------------------------------------------------------------------
data$title = sapply(data$name, FUN=function(x) { strsplit(x, split='[,.]')[[1]][2]})
data$title = sub(' ', '', data$title)
table(data$title)
data$title[data$title %in% c('Mme', 'Mlle', 'Ms')] <- 'Miss'
data$title[data$title %in% c('Capt', 'Col', 'Don', 'Major', 'Sir', 'Dr')] <- 'Sir'
data$title[data$title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
table(data$title)
data$title = factor(data$title)
str(data$title)
sum(is.na(data$title)) # no NAs

# sex: more males on board than females, but ok?
# ----------------------------------------------------------------------------
str(data$sex)
tab = table(data$sex)
ptab = prop.table(tab)
barplot(ptab)
sum(is.na(data$sex)) # no NAs

# familysize: combine all family members into single count
# todo: make factor ?
# ----------------------------------------------------------------------------
data$familysize = data$sibsp + data$parch + 1
#data$familysizefac = cut(data$familysize, breaks=c(1,2,4,max(data$familysize)), include.lowest=T, right=F)
familysizefac = cut2(data$familysize, g=3, minmax=T)
data$familysizefac = ordered(familysizefac, labels=c("single", "small", "big"))
barplot(prop.table(table(data$familysizefac))) 
data$familysizefac
#mosaicplot(data$familysizefac ~ data$survived, color=T)

# age:
# ----------------------------------------------------------------------------
str(data$age)
hist(data$age,40) # many with 0 years?
sum(is.na(data$age)) # many NAs!
data$age==0 # many NAs!
data$age[!is.na(data$age)] # Younger than encoded in decimal 0.x
data$age[(!is.na(data$age)) & data$age <= 1] # Many infants?

# Fit a decision tree to fill in missing ages
summary(data$age)
agefit = rpart(age ~ pclass + sex + sibsp + parch + fare + embarked + title + familysize, data=data[!is.na(data$age),], method="anova")
data$age[is.na(data$age)] = predict(agefit, data[is.na(data$age), ])
summary(data$age) # pretty similar overall statistic, which is good

# Create a new "child" variable
data$child = factor(data$age < 16)
str(data$child) 

# ticket: almost unique identifier, some people must have travelled with same ticket
# ----------------------------------------------------------------------------
str(data$ticket)
head(data$ticket, 50)
sum(is.na(data$ticket)) # no NAs
sum(data$ticket == '') # no NAs
length(unique(data$ticket))

# fare: 17 "missing" values (fare=0): use decision tree again to fill in?
# Or maybe it reflects a useful property. Almost all case have perished...
# Still, exp. distributed. Try as factor
# ----------------------------------------------------------------------------
str(data$fare)
sum(is.na(data$fare)) # One NA!
nai = which(is.na(data$fare))
data$fare[nai] = median(data$fare, na.rm=T)
hist(data$fare,50)
hist(log(data$fare),50)
u = unique(data$fare)
u[order(u)]
sum(data$fare == 0) 
hist(data$fare[data$fare!=0],50)

#farecat = cut(data$fare, breaks=c(min(data$fare),8,16,32,64,max(data$fare)), include.lowest=T, labels=F)
#farecat = ordered(cut(data$fare, breaks=c(min(data$fare),8,16,32,64,max(data$fare)), include.lowest=T, labels=F))
numfarecats = 3
farecat = cut2(data$fare, g=numfarecats, minmax=T)
farecat = ordered(farecat, labels=c(1:numfarecats))
barplot(prop.table(table(farecat)))
sum(is.na(farecat))
data$farecat = farecat

# cabin: most cabin fields are empty. Hmmm
# ----------------------------------------------------------------------------
str(data$cabin)
sum(is.na(data$cabin)) # 
sum(data$cabin == '') 

data$cabcat = factor(sapply(data$cabin, FUN=function(x) { substr(strsplit(x, " ")[[1]][1], 1, 1) } ))
data$cabeven = factor(sapply(data$cabin, FUN=function(x) { as.numeric(substr(strsplit(x, " ")[[1]][1], 2, 10)) %% 2 == 0 } ))

# extend to family members
familyid = paste(as.character(data$familysize), data$surname, sep="")
#assigned = data[!is.na(data$cabcat)]
#assnames = unique(assigned$surname)
assignedI = which(!is.na(data$cabcat))
assignedid = unique(familyid[assignedI])

findUnassigned = function(famid){
  members = which(familyid==famid)
  family = data[members,]
  isfam = nrow(family) > 1 & length(unique(family$pclass))==1 & length(unique(family$embarked))==1
  unassigned = sum(is.na(family$cabcat)) > 0
  
  if(isfam & unassigned){
    print(paste(famid))
  }
  return
}

unass = sapply(assignedid, FUN=findUnassigned)

# embarked: most cabin fields are empty. Hmmm
# ----------------------------------------------------------------------------
str(data$embarked) # One level has empty label ''
nai = which(data$embarked == '')
sum(is.na(data$embarked)) # no NAs
tab = table(data$embarked)
ptab = prop.table(tab)
barplot(ptab)
data$embarked[nai] = 'S'
data$embarked = factor(data$embarked)
barplot(prop.table(table(data$embarked)))

# Create a factor variables identifying large families uniquely
# ----------------------------------------------------------------------------
sizecut = 4
data$surname = sapply(data$name, FUN=function(x) { strsplit(x, split='[,.]')[[1]][1]})
length(unique(data$surname))
data$familyid = paste(as.character(data$familysize), data$surname, sep="")
data$familyid[data$familysize <= sizecut] = 'Small'
length(unique(data$familyid)) # not totally unique but well...

# still small families not labelled as such, perhaps because of different surnames
smallfams = data.frame(table(data$familyid))
smallfams = smallfams[smallfams$Freq <= sizecut, ]
data$familyid[data$familyid %in% smallfams$Var1] = "Small"
length(unique(data$familyid)) # not totally unique but well...
table(data$familyid)
data$familyid = factor(data$familyid)
str(data$familyid)

# Remove useless variables
# ----------------------------------------------------------------------------
drop = c("name", "sibsp", "parch", "ticket", "cabin", "cabcat", "cabeven", "surname")
data = data[, !names(data) %in% drop]

# Split back into train and test set
# ----------------------------------------------------------------------------
train = data[1:ntr,]
test = tail(data, nte)
nrow(train) == ntr
nrow(test) == nte

str(train)

# Fix up factors? Levels etc...
# ----------------------------------------------------------------------------
nearZeroVar(train, saveMetrics=T)

# Confounders, correlated predictors
# ----------------------------------------------------------------------------
which(is.na(train))
#corMat = cor(train)
#findLinearCombos(train)


save(train, test, file="data/mungedData.RData")
