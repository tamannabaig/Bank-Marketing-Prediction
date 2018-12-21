#Author: Tamanna Baig
#Date: December 7, 2018
#Purpose: To assess Bank Marketing dataset and predict if the customer subscribed a deposit or not

#Remove all files from the environment
rm(list = ls())

#Import necessary libraries
library(caret)
library(randomForest)
library(mlbench)
library(mice)
library(caTools)


#Loading the Dataset
dataset <- read.csv('bank-full.csv', header = TRUE, sep=";", na.strings = "?")

#View dataset that consists of 45211 observations and 17 variables
View(dataset)

#Lists all the variable names
names(dataset)

#Results in the count of observations and variables
dim(dataset)

#To DIsplay Column names and their datatypes
str(dataset)

#Summary of the Dataset
summary(dataset)

#Checking for Missing Values in the datasey
missing_ds <- subset(dataset, select = -c(y))
summary(missing_ds)

#Tabular view of missing values with respect to the columns they are present in
md.pattern(missing_ds)

#*********************There is no missing data*********************#

#Plotting the predictors against the response to better understand their relationship
qplot(dataset$y, dataset$age, data=dataset, geom="boxplot", xlab="Subscription", ylab="Age")
qplot(dataset$y, dataset$balance, data=dataset, geom="boxplot", xlab="Subscription", ylab="Balance")
qplot(dataset$y, dataset$duration, data=dataset, geom="boxplot", xlab="Subscription", ylab="Duration of call")

#View the Summary of the dataset that provides details about min, max, median, mean of each variable
summary(dataset)

#Splitting the Dataset for Model Selection:
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Look for the proportion of outcome categories that is: Yes and No
prop.table(table(training_set$y))
prop.table(table(test_set$y))

######################################################################################
#Model1 : Logistic Regression

logit <- glm(formula = y ~.,
                  family = binomial("logit"),
                  data = training_set)

#Predicting Logistic Regression classifier
y_pred1 = predict(logit, newdata = test_set[-17])
y_pred1

# Making the Confusion Matrix for Logistic Regression
cm1 =  prop.table(table(test_set[, 17], y_pred1 > 0.5))
cm1

#The Accuracy of the Logistic Regression Classifier
accuracy1 <- (cm1[1,1] + cm1[2,2])/(cm1[1,1] + cm1[2,2]+ cm1[1,2]+ cm1[2,1])
accuracy1
#The precision of the Logistic Regression Classifier (how often it is correct)
precision1 <- (cm1[2,2])/(cm1[2,2]+cm1[1,2])
precision1
#Recall/sensitivity/True positive rate of the Logistic Regression Classifier
recall1 <- cm1[2,2]/(cm1[2,2]+cm1[2,1])
recall1

######################################################################################
#Model 2 : Random Forest
#To call Random Forest classifier
forest = randomForest(x = training_set[-c(17)],
                      y = training_set$y,
                      ntree = 25)

plot(forest)
#Plotting to analyse which variable has the highest importance
varImpPlot(forest, main = "Predictor Relevance")

#Predicting the Random Forest classifier 
y_pred2 = predict(forest, test_set, type = "class")


#Confusion Matrix of the Random Forest Classifier
cm2 <- prop.table(table(y_pred2, test_set$y))
cm2

#The Accuracy of the Random Forest Classifier
accuracy2 <- (cm2[1,1] + cm2[2,2])/(cm2[1,1] + cm2[2,2]+ cm2[1,2]+ cm2[2,1])
accuracy2

#The precision of the Random Forest Classifier (how often it is correct)
precision2 <- (cm2[2,2])/(cm2[2,2]+cm2[1,2])
precision2

#Recall/sensitivity/True positive rate of the Random Forest Classifier
recall2 <- cm2[2,2]/(cm2[2,2]+cm2[2,1])
recall2
