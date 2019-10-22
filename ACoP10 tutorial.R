###############################################################
###############################################################
###############################################################
###                                                         ###
###     Application of Machine Learning                     ###
###     Hands on Use case with ReseqTB database.            ###
###     The full dataset is publicly available by request   ###
###     on https://platform.reseqtb.org/                    ###
###                                                         ###
###     The purpose of the database is to link mutations    ###
###     within various genes of different strans of         ###
###     tuberculosis isolates to the phenotype of drug      ###
###     susceptibility or resistance.                       ###
###                                                         ###
###     This code analyzes the number of mutations          ###
###     within different genes and builds a classification  ###
###     model to predict the resistance or                  ###
###     susceptibility of an isolate to a Drug A. The data  ###
###     used for this tutorial is simulated, but still      ###
###     resembles the actual data.  Gene names have been    ###
###     masked, but interested individuals can pursue       ###
###     further investigation via a download of the full    ###
###     database.                                           ###
###                                                         ###
###                                                         ###
###############################################################
###############################################################
###############################################################


#Clear R environment

rm(list=ls())


### Installation of required packages

is.installed <- function(mypkg) { is.element(mypkg, installed.packages()[,1]) }

if (is.installed("data.table") == FALSE) {install.packages("data.table")} 

if (is.installed("caret") == FALSE) {install.packages("caret")} 

if (is.installed("pROC") == FALSE) {install.packages("pROC")} 

if (is.installed("sampling") == FALSE) {install.packages("sampling")} 

if (is.installed("ROCR") == FALSE) {install.packages("ROCR")} 

if (is.installed("ggplot2") == FALSE) {install.packages("ggplot2")} 

if (is.installed("e1071") == FALSE) {install.packages("e1071")} 

if (is.installed("gbm") == FALSE) {install.packages("gbm")} 

### Required packages

library(data.table)

library(caret) 

library(pROC)

library(sampling)

library(ROCR)

library(ggplot2)


### Set working directory

#Please set the working directory to the folder you will store the data
#setwd("")


### Read in data set

data.set <- "ACoP10 tutorial simulated dataset - Drug A.csv"

dataset <- as.data.frame(fread(input = data.set, check.names = TRUE)) 

dim(dataset)
#[1] 3219 2113

View(head(dataset))

dataset <- dataset[ , -c(1)]

dataset$Drug.Susceptibility.Testing <- as.factor(dataset$Drug.Susceptibility.Testing)

plot(dataset$Drug.Susceptibility.Testing)

ntrain <- floor(min(as.data.frame(table(dataset$Drug.Susceptibility.Testing))$Freq)*0.8)


### Create a balanced training set ###

## Create the random sample for the training set
set.seed(1)

random_sample <- strata(data = dataset, 
                        
                        stratanames = "Drug.Susceptibility.Testing", 
                        
                        size = c(ntrain , ntrain), 
                        
                        method = "srswor" )# simple random sampling without replacement

## Extract the training set
training_set <- dataset[random_sample$ID_unit, ]

## Preprocessing of the training set set

# Columns containing non-available values for at least one isolate are removed
training_set <- training_set[ , apply(training_set, 2, function(x) !any(is.na(x)))]

# Columns displaying no variance (sd = 0) or almost no variance are indexed.
index_low_variance <- nearZeroVar(x = training_set)

# Remove the previously indexed columns
training_set <- training_set[ , -index_low_variance] 

dim(training_set)
#[1] 1926 1462

### Create the test set
testing_set <- dataset[-random_sample$ID_unit, ]

testing_set <- subset(testing_set, select = names(training_set))


### Remove correlated features

# calculate correlation matrix
correlationMatrix <- cor(training_set[,-1])

# find attributes that are highly corrected (ideally >0.5)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5 , names = TRUE)

?findCorrelation #to find more information about "findCorrelation" function

# Create a training set without correlated features
training_set_reduced <- training_set[, !colnames(training_set) %in% highlyCorrelated]

# Create a test set without correlated features
testing_set_reduced <- testing_set[,!colnames(testing_set) %in% highlyCorrelated]


##################################

### GLM - Logistic regression  ###

##################################

## Training the model

set.seed(1)

?train

# http://topepo.github.io/caret/available-models.html # Available Models

ptm <- proc.time()

glmFit <- train(Drug.Susceptibility.Testing  ~ .,#A formula of the form y ~ x1 + x2 + ...
                
                data = training_set_reduced,#Data frame from which variables in formula are going to be taken.
                
                metric = "Accuracy",#A string that specifies what summary metric will be used to select the optimal model. 
                
                trControl = trainControl(method = "none"),#A list of values that define how this function acts. 
                
                method = "glm",#A string specifying which classification or regression model to use
                
                family = "binomial")
                
proc.time() - ptm


### Model and performance analysis

glmFit

plot(glmFit)

plot(varImp(glmFit), top=20)

predictions.train <- predict(glmFit, newdata = training_set_reduced , type = "prob" , na.action = na.pass) 

auc.training <- auc(roc(predictor = predictions.train$SUSCEPTIBLE , response = training_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Training set")) 

predictions.test <- predict(glmFit, newdata = testing_set_reduced , type="prob" , na.action = na.pass)  

auc.test <- auc(roc(predictor= predictions.test$SUSCEPTIBLE , response = testing_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Test set")) 

results.glm <- list("GLM model", glmFit , "AUC ROC Training" , auc.training, "AUC ROC Test", auc.test) 

results.glm

confusionMatrix(  predict(glmFit, newdata = training_set_reduced) , training_set_reduced$Drug.Susceptibility.Testing)

confusionMatrix(  predict(glmFit, newdata = testing_set_reduced) , testing_set_reduced$Drug.Susceptibility.Testing)



#############

### K-nn  ###

#############

## Training the model

set.seed(1)

ptm <- proc.time()

knnFit <- train(Drug.Susceptibility.Testing  ~ .,#A formula of the form y ~ x1 + x2 + ...
                
                data = training_set_reduced,#Data frame from which variables in formula are going to be taken.
                
                metric = "Accuracy",#A string that specifies what summary metric will be used to select the optimal model. 
                
                trControl = trainControl(method = "none"),#A list of values that define how this function acts. 
                
                method = "knn", #A string specifying which classification or regression model to use
                
                preProcess = c("center","scale")) #A string vector that defines a pre-processing of the predictor data. 
                
                
proc.time() - ptm


### Model and performance analysis

knnFit

knnFit$finalModel

plot(knnFit)

plot(varImp(knnFit), top=20)

predictions.train <- predict(knnFit, newdata = training_set_reduced , type = "prob" , na.action = na.pass) 

auc.training <- auc(roc(predictor = predictions.train$SUSCEPTIBLE , response = training_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Training set")) 

predictions.test <- predict(knnFit, newdata = testing_set_reduced , type="prob" , na.action = na.pass)  

auc.test <- auc(roc(predictor= predictions.test$SUSCEPTIBLE , response = testing_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Test set")) 

results.knn <- list("Knn model", knnFit , "AUC ROC Training" , auc.training, "AUC ROC Test", auc.test) 

results.knn

confusionMatrix(  predict(knnFit, newdata = training_set_reduced) , training_set_reduced$Drug.Susceptibility.Testing)

confusionMatrix(  predict(knnFit, newdata = testing_set_reduced) , testing_set_reduced$Drug.Susceptibility.Testing)


#############

### K-nn

### With tuning parameters

#############


## Cross Validation setting

set.seed(1)

?trainControl

ctrl <- trainControl(method = "cv",#The resampling method
                     
                     allowParallel = FALSE,#If a parallel backend is available, should the function use it?
                     
                     verboseIter = TRUE,#A logical for printing a training log.
                     
                     number = 5,#The number of folds 
                     
                     #repeats = 1,#The number of complete sets of folds to compute
                     
                     classProbs = TRUE# should class probabilities be computed for each resample?
)


## Setting the hyperparameters search

?expand.grid

knnGrid <-  expand.grid(  
  
  k = c(1,3,5,7,9) ) 


## Training the model

#Fit Predictive Models over Different Tuning Parameters

ptm <- proc.time()

knnFit_tuning <- train(Drug.Susceptibility.Testing  ~ .,#A formula of the form y ~ x1 + x2 + ...
                
                data = training_set_reduced, ## le digo cuales son mis datos para armar el modelo
                
                metric = "Accuracy",#A string that specifies what summary metric will be used to select the optimal model. 
                
                method = "knn", #A string specifying which classification or regression model to use
                
                trControl = ctrl,#A list of values that define how this function acts. 
                
                preProcess = c("center","scale"), #A string vector that defines a pre-processing of the predictor data. 
                
                tuneGrid = knnGrid )#A data frame with possible tuning values.

proc.time() - ptm


### Model and performance analysis

knnFit_tuning

plot(knnFit_tuning)

plot(varImp(knnFit_tuning), top=20)

predictions.train <- predict(knnFit_tuning, newdata = training_set_reduced , type = "prob" , na.action = na.pass) 

auc.training <- auc(roc(predictor = predictions.train$SUSCEPTIBLE , response = training_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Training set"))

predictions.test <- predict(knnFit_tuning, newdata = testing_set_reduced , type="prob" , na.action = na.pass)  

auc.test <- auc(roc(predictor= predictions.test$SUSCEPTIBLE , response = testing_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Test set")) 

results.knn_tuning <- list("Knn model with tuning parameters", knnFit , "AUC ROC Training" , auc.training, "AUC ROC Test", auc.test) 

results.knn_tuning

confusionMatrix(  predict(knnFit_tuning, newdata = training_set_reduced) , training_set_reduced$Drug.Susceptibility.Testing)

confusionMatrix(  predict(knnFit_tuning, newdata = testing_set_reduced) , testing_set_reduced$Drug.Susceptibility.Testing)





#############

### GBM   ###

#############




## Cross Validation setting

set.seed(1)

ctrl <- trainControl(method = "cv",#The resampling method
                        
                        allowParallel = FALSE,#If a parallel backend is available, should the function use it?
                        
                        verboseIter = TRUE,#A logical for printing a training log.
                        
                        number = 5,#The number of folds 
                        
                        repeats = 1,#The number of complete sets of folds to compute
                        
                        classProbs = TRUE# should class probabilities be computed for each resample?
)


## Setting the hyperparameters search

gbmGrid <-  expand.grid(  
  
  interaction.depth = 1 , #	Integer specifying the maximum depth of each tree (i.e., the highest level of variable interactions allowed). 
  
  n.trees = seq(from =  0 , to = 2000 , by = 100), # Integer specifying the total number of trees to fit.
  
  shrinkage = 0.01 , # A shrinkage parameter applied to each tree in the expansion. Also known as the learning rate or step-size reduction; 
  
  n.minobsinnode = 10 ) # Integer specifying the minimum number of observations in the terminal nodes of the trees. 


## Training the model with reduced training and test set

#Fit Predictive Models over Different Tuning Parameters

ptm <- proc.time()

gbmfit_reduced <- train( Drug.Susceptibility.Testing ~ .,#A formula of the form y ~ x1 + x2 + ...
                         
                         data = training_set_reduced,#Data frame from which variables in formula are going to be taken.
                         
                         metric = "Accuracy",#A string that specifies what summary metric will be used 
                         #to select the optimal model. 
                         
                         method = "gbm",#A string specifying which classification or regression model to use.
                         
                         trControl = ctrl,#A list of values that define how this function acts. 
                         
                         tuneGrid = gbmGrid ,#A data frame with possible tuning values.
                         
                         distribution = "bernoulli" ) #A character string specifying the name of the distribution to use 

proc.time() - ptm


### Model and performance analysis

gbmfit_reduced

plot(gbmfit_reduced) 

importance.variables <- head(summary(gbmfit_reduced), n = 20) 

importance.variables <- transform(importance.variables, var = reorder(var, rel.inf)) 

ggplot(data = importance.variables , aes(x = var, y = rel.inf)) + geom_bar(stat="identity", fill="steelblue")  + coord_flip() + theme_minimal() + labs(title = "20 most influence variables in GBM", y = "Relative Influence" , x = "Variable") + theme(plot.title = element_text(hjust = 0.5))  

library(pROC) 

predictions.train <- predict(gbmfit_reduced, newdata = training_set_reduced , type = "prob" , na.action = na.pass) 

auc.training <- auc(roc(predictor = predictions.train$SUSCEPTIBLE , response = training_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Training set")) 

predictions.test <- predict(gbmfit_reduced, newdata = testing_set_reduced , type="prob" , na.action = na.pass)  

auc.test <- auc(roc(predictor= predictions.test$SUSCEPTIBLE , response = testing_set_reduced$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Test set")) 

results.gbm_reduced <- list("GBM model", gbmfit_reduced , "AUC ROC Training" , auc.training, "AUC ROC Test", auc.test) 

results.gbm_reduced

confusionMatrix(  predict(gbmfit_reduced, newdata = training_set_reduced) , training_set_reduced$Drug.Susceptibility.Testing)

confusionMatrix(  predict(gbmfit_reduced, newdata = testing_set_reduced) , testing_set_reduced$Drug.Susceptibility.Testing)



## Training the model

#Fit Predictive Models over Different Tuning Parameters

ptm <- proc.time()

gbmfit <- train( Drug.Susceptibility.Testing ~ .,#A formula of the form y ~ x1 + x2 + ...
             
             data = training_set,#Data frame from which variables in formula are going to be taken.
             
             metric = "Accuracy",#A string that specifies what summary metric will be used to select the optimal model. 
             
             method = "gbm",#A string specifying which classification or regression model to use.
             
             trControl = ctrl,#A list of values that define how this function acts. 
             
             tuneGrid = gbmGrid ,#A data frame with possible tuning values.
             
             distribution = "bernoulli" ) #A character string specifying the name of the distribution to use 

proc.time() - ptm


### Model and performance analysis

gbmfit 

plot(gbmfit) 

importance.variables <- head(summary(gbmfit), n = 20) 

importance.variables <- transform(importance.variables, var = reorder(var, rel.inf)) 

ggplot(data = importance.variables , aes(x = var, y = rel.inf)) + geom_bar(stat="identity", fill="steelblue")  + coord_flip() + theme_minimal() + labs(title = "20 most influence variables in Boosting", y = "Relative Influence" , x = "Variable") + theme(plot.title = element_text(hjust = 0.5))  

library(pROC) 

predictions.train <- predict(gbmfit, newdata = training_set , type = "prob" , na.action = na.pass) 

auc.training <- auc(roc(predictor = predictions.train$SUSCEPTIBLE , response = training_set$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Training set")) 

predictions.test <- predict(gbmfit, newdata = testing_set , type="prob" , na.action = na.pass)  

auc.test <- auc(roc(predictor= predictions.test$SUSCEPTIBLE , response = testing_set$Drug.Susceptibility.Testing , direction = "<", plot = TRUE, main ="ROC Test set")) 

results.gbm <- list("GBM model - all features", gbmfit , "AUC ROC Training" , auc.training, "AUC ROC Test", auc.test) 

results.gbm

confusionMatrix(  predict(gbmfit, newdata = training_set) , training_set$Drug.Susceptibility.Testing)

confusionMatrix(  predict(gbmfit, newdata = testing_set) , testing_set$Drug.Susceptibility.Testing)

















