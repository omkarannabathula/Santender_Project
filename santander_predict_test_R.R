rm(list=ls())

setwd('F:/Edvisor Project/Santender_Project')
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)


#library(ff)

#reading data

tn = read.csv('train.csv', header= T)
test = read.csv('test.csv', header= T)

tn = tn[,-c(1)]

str(tn$target)


tn$target = as.factor(tn$target)

#summarise the data

summarizeColumns(tn)

summary(tn)

#distribution of target class

barplot(prop.table(table(tn$target)),
        
        col= rainbow(2),
        ylim= c(0, 1),
        main= 'Class distribution')

#Missing value analysis

missing_val = data.frame(apply(tn,2,function(x){sum(is.na(x))}))

#covert row names to columns

missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL

#renaming variable name

names(missing_val)[1] = 'Missing values'

#re-arrange column

missing_val = missing_val[c(2,1)]

#write the output

write.csv(missing_val, 'Missing_valuesR.csv', row.names = F)

#taking undersampling as the data is huge and will not be processed 

#statrified sampling

table(tn$target)

install.packages('dplyr')


library(sampling)

#strata = strata(tn, c('target'), size = c(20000, 20000), method = 'srswor')

#strata = strata(tn, c('target'), size = c(5000, 5000), method = 'srswor')

strata = strata(tn, c('target'), size = c(2000, 2000), method = 'srswor')

tn.strata = getdata(tn, strata)

tn = tn.strata[, -c(202:204)]

install.packages("magrittr")
library(magrittr)
library(dplyr)

tn = tn%>%select(target, everything())

#recols = c('ID_code', 'target')
#tn = tn[, c(recols, setdiff(names(tn), recols))]
# 
# tn$target = as.factor(tn$target)
# tn$ID_code = as.numeric_version(tn$ID_code)

#tn = tn[c(2,1)]

#rm(hacide.test, hacide.train)

#Outlier Analysis

cnames = tn[,-c(1)]

# cnames1 = cnames[,c(1:20)]
# 
# cnames2 = cnames[,c(21:40)]
# 
# cnames3 = cnames[,c(41:60)]

colnames(cnames)

install.packages("mlr")
library(mlr)

outliers = c()

#Loop through the list of columns
for(i in cnames){

   # Get the Min/Max values
   Tmax = quantile(tn[,i],0.75, na.rm=TRUE) + (IQR(tn[,i], na.rm=TRUE) * 1.5 )
   Tmin = quantile(tn[,i],0.25, na.rm=TRUE) - (IQR(tn[,i], na.rm=TRUE) * 1.5 )
   
   idx = which(tn[,i] < Tmin | tn[,i] > Tmax)
   
   paste(i, length(idx), sep = '')
   
   outliers = c(outliers, idx)
   
}

tn = tn[-outliers]

##Feature Scaling

#Normalisation

# install.packages("sqldf")
# library(sqldf)
cnames = as.data.frame(cnames)
cnames = as.numeric(cnames)

class(cnames)

for(i in cnames){
   
   tn[,i] = (tn[,i] - min(tn[,i]))/(max(tn[,i] - min(tn[,i])))
}

# normalize_column = function (data, column) {
# 
#   temp_col = data[[column]]
# 
#   tmin = min(temp_col, na.rm=True)
#   tmax = max(temp_col, na.rm=True)
# 
#   temp_output = (temp_col - tmin)/(tmax - tmin)
# 
#   return(temp_output)
# }
# 
# for (i in cnames){
# 
#   tn = normalize_column(tn, i)
# }


#Feature selection

# install.packages('Boruta')
# library(ranger)
# library (Boruta)
# library(mlbench)
# 
# set.seed(111)
# 
# boruta = Boruta(target ~., data = tn, maxRuns= 500)
# 
# print(boruta)
# plot(boruta, las=2, cex.axis=0.3)
# 
# #tentative fix
# 
# bor = TentativeRoughFix(boruta)
# print(bor)
# 
# attStats(boruta)

#model development

set.seed(1234)
tn.index = sample(nrow(tn), 3000, replace = F)

install.packages('sandwich')
library(list)
train = tn[tn.index,]
test= tn[-tn.index,]

library(randomForest)
rfAll = randomForest(target~., train, importance= TRUE, ntress= 500)

#extract rules
library(inTrees)

tree = RF2List(rfAll)

rules = extractRules(tree, tn[,-1])

#view rules

rules[1:2,]

#rule metrics

ruleMetric = getRuleMetric(rules, tn[,-1], tn$target)

ruleMetric[1:2,]

#predicting test data

RF_predict = predict(rfAll, test[,-1])

#evaluate
library(caret)

confmatrix = table(test$target, RF_predict)
confusionMatrix(confmatrix)

#FNR

#FNR = FN/FN+TP

110/(110+380)

#Recall = 69.7

#Precision= 75.05

#Accuracy = 74.4
#FNR = 22

#logistic Regression

logistic = glm(target~., train, family= 'binomial')

#summary

summary(logistic)

#predict

logpredict = predict(logistic, newdata= test, type = 'response')

logpredict= ifelse(logpredict > 0.5, 1, 0)

#evaluate

conflogistic = table(test$target, logpredict)

#Accuracy

(392+390)/1000

#FNR

100/(100+390)

#Accuracy = 78.2
#FNR = 20.4
#Recall= 70.56
#Precision= 75.10


#KNN imputation

library(class)

#predict test and train data

knnpredict = knn(train[,2:201], test[,2:201], train$target, k= 3)

confknn = table(knnpredict, test$target)

#Accuracy

sum(diag(confknn)/nrow(test))

#FNR

123/(123+170)

#Accuracy
# k = 1 -- 52.8
# k = 3 -- 57.1
# k = 5 -- 55.4
# k = 7 -- 55.7

#FNR = 41.9
#Recall= 59.3
#Precision= 30.79

#Naive bayes

library(e1071)

#implement

NBayes = naiveBayes(target ~., data = train)

#predict using test cases

NBpredict = predict(NBayes, test[,2:201], type = 'class')

#confusion matrix

confNB = table(observed = test[,1], predicted = NBpredict)
confusionMatrix(confNB)

#FNR

100/(100+390)

#Accuracy = 79.6
#FNR = 20.4
#Recall= 72.5
#Precision= 77.8

#test['target'] = NBpredict(test)

test$target <- as.factor(test$target)
levels(test$target)[levels(test$target) == 0] <- 'Yes'
levels(test$target)[levels(test$target) == 1] <- 'No'

write.csv(test, file = "santander_test_predict_R.csv")
