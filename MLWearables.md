---
title: "Classification of Activities Using Wearable Electronics Metrics"
author: "Kyle Hayes"
date: "May 8, 2018"
output: 
  html_document:
    keep_md: true
self_contained: true
---



# Introduction

The purpose of this exploration is to use Machine Learning to predict the manner in which a person doing an excercise completed the particular excercise.  This classification variable, known as "classe" will be predicted using feedback from accelerometer data attached to the individual.

# Data Preparation and Cleaning

First the training data is downloaded and read into R.  Note that both NA and blank must be designated as missing values.


```r
LinkURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

download.file(LinkURL,"Wear_Training.csv")

training <- read.csv("Wear_Training.csv", na.string=c("","NA"))
```

Useful libraries for the excercise are loaded and variables with majority missing values are removed.  This was done in lieu of an imputation methodology to improve processing time and because there appeared to be sufficient variables remaining after removal to construct a suitable model. 

 

```r
library(caret)
library(dplyr)
library(parallel)
library(doParallel)

set.seed(3456)

#Check for missing values
miss_cols <- which(sapply(training,function(x) any(is.na(x))))

#drop columns with missing values
training <- training[,-miss_cols]
```

The training set is then split into a training (75%) and hold out sample (25%).


```r
inTrain = createDataPartition(training$classe, p = 3/4)[[1]]
train_dev <- training[inTrain,]
test_dev <- training[-inTrain,]
```

# Model Build

The model is built on the train_dev set and then validated on the test_dev set for accuracy.  We make use of R's parallel processing capabilities in order to improve processing time.  This allows us to use the extremely effective Random Forest approach rather than a less effective, but computationally more convenient approach like Linear Discriminant Analysis. 


```r
#Set up for Parallel Processing
cluster <- makeCluster(detectCores() -1)
registerDoParallel(cluster)

#Set up Control for fit
fitControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
```

# Cross Validation
In the above code,  we adjust the resampling method from bootstrapping to k-fold cross validation with 5 folds. This means we are no longer using leave-one-out sampling but instead creating 5 random samples with which to train the model.  This significantly improves processing speed during the model fit process.


```r
mod1 <- train(classe~.,method="rf",trControl=fitControl, data=train_dev)
```

# Expected Error Rate
If we examine a confusion matrix of the 5 fold cross voalidated model, we get an accurracy of .999.

```r
#first deregister cluster
stopCluster(cluster)
registerDoSEQ()
#Generate 
confusionMatrix(mod1)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.0  0.0
##          D  0.0  0.0  0.0 16.4  0.0
##          E  0.0  0.0  0.0  0.0 18.4
##                             
##  Accuracy (average) : 0.9998
```
 
Also, if we apply our model to our holdout sample and create a confusion matrix we get an accuracy of .999.


```r
pred1 <- predict(mod1,test_dev)
table(pred1,test_dev$classe)
```

```
##      
## pred1    A    B    C    D    E
##     A 1395    0    0    0    0
##     B    0  949    0    0    0
##     C    0    0  855    0    0
##     D    0    0    0  804    0
##     E    0    0    0    0  901
```

As such, we expect the model to be extremely accurate when applied to new, similar data.

# Justification of Choices
The majority of choices were the result of gains in computational efficiency.  Random Forests are known for being excellent in classification problems, especially when there are a large number of predictors.  Random Forests also require minimal preprocessing.  Finally, the performance of the model selected was such that it was not necessary to test other methods for improved accuracy.
