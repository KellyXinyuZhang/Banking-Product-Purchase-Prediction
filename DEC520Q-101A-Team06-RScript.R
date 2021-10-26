install.packages('fastDummies')
library('fastDummies')
library('readr')
bank_full <- read_delim("bank-full.csv",";", escape_double = FALSE, trim_ws = TRUE)
dataF <- bank_full
dataF$y <- ifelse(dataF$y=="no",0,1)
###EXPLORATORY DATA ANALYSIS
library(corrplot)
corrplot(cor(dataF), method = "square", type = "lower", order="hclust", sig.level = 0.01)
plot(factor(y) ~ factor(age), data=dataF, col=c(8,2),  xlab="Age", ylab="Success", main="Figure 1 : Success by Age")
plot(factor(y) ~ factor(job), data=dataF, col=c(8,2), xlab="Job Type", ylab="Success", main="Figure 2 : Success by Jobs")
plot(factor(y) ~ factor(marital), data=dataF, col=c(8,2), xlab="Marital Status", ylab="Success", main="Figure 3 : Success by Marital Status")
plot(factor(y) ~ factor(education), data=dataF, col=c(8,2), xlab="Education", ylab="Success", main="Figure 4 : Success by Education")
plot(factor(y) ~ factor(default), data=dataF, col=c(8,2), xlab="Previous Defualt", ylab="Success", main="Figure 5 : Success by Previous Defualt")
plot(factor(y) ~ factor(loan), data=dataF, col=c(8,2),  xlab="Personal Loan", ylab="Success", main="Figure 6 : Success by Personal Loan") 
plot(factor(y) ~ factor(housing), data=dataF, col=c(8,2),  xlab="Housing Loan", ylab="Success", main="Figure 7 : Success by Housing Loan")
plot(factor(y) ~ factor(month), data=dataF, col=c(8,2), xlab="Month", ylab="Success", main="Figure 8 : Success by Month")
plot(factor(y) ~ factor(campaign), data=dataF, col=c(8,2), xlab="Number of Contacts", ylab="Success", main="Figure 9 : Success by Contact Rate")
plot(factor(y) ~ factor(contact), data=dataF, col=c(8,2), xlab="Contact Method", ylab="Success", main="Figure 10 : Success by Contact Method")
plot(factor(y) ~ factor(poutcome), data=dataF, col=c(8,2), xlab="Previous Outcome", ylab="Success", main="Figure 11 : Success by Previous Outcome")
bank_cor <- select_if(dataF, is.numeric) %>% cor()
corrplot(bank_cor, method = "number")
#DATA PREPARATION
dataF$education <- ifelse(dataF$education=="unknown",0,ifelse(dataF$education=="primary",1,ifelse(dataF$education=="secondary",2,3)))
dataF$default <- ifelse(dataF$default=="no",0,1)
dataF$housing <- ifelse(dataF$housing=="no",0,1)
dataF$loan <- ifelse(dataF$loan=="no",0,1)
dataF$pdays <- ifelse(dataF$pdays==-1,0,1)
dataF$month <- ifelse(dataF$month=="jan"|dataF$month=="feb"|dataF$month=="mar",1,ifelse(dataF$month=="apr"|dataF$month=="may"|dataF$month=="jun",2,ifelse(dataF$month=="jul"|dataF$month=="aug"|dataF$month=="sep",3,4)))
unique(dataF$month)
dataF <- dummy_cols(dataF, select_columns = c('job','marital','contact','poutcome'), remove_selected_columns = TRUE)
dataF$job_blue_collar<-dataF$`job_blue-collar`
dataF$`job_blue-collar` <- NULL
dataF$job_self_employed<-dataF$`job_self-employed`
dataF$`job_self-employed`<- NULL
dataF$job_admin <- dataF$job_admin.
dataF$job_admin. <- NULL
dataF = subset(dataF, select = -c(job_unknown,marital_single,contact_unknown,poutcome_unknown) )
head(dataF)
#remove duration since highly correlated 
#also we dont know duration of call before call occurs
dataF$duration<- NULL
dataF$job_blue_collar  <- as.factor(dataF$job_blue_collar)
dataF$job_self_employed<- as.factor(dataF$job_self_employed)
dataF$education <- as.factor(dataF$education)
dataF$default <- as.factor(dataF$default)
dataF$housing <- as.factor(dataF$housing)
dataF$loan <- as.factor(dataF$loan)
dataF$campaign <- as.factor(dataF$campaign)
dataF$pdays <- as.factor(dataF$pdays)
dataF$y <- as.factor(dataF$y)
dataF$job_entrepreneur <- as.factor(dataF$job_entrepreneur)
dataF$job_housemaid <- as.factor(dataF$job_housemaid)
dataF$job_management <- as.factor(dataF$job_management)
dataF$job_retired <- as.factor(dataF$job_retired)
dataF$job_services <- as.factor(dataF$job_services)
dataF$job_student <- as.factor(dataF$job_student)
dataF$job_technician <- as.factor(dataF$job_technician)
dataF$job_unemployed <- as.factor(dataF$job_unemployed)
dataF$marital_divorced <- as.factor(dataF$marital_divorced)
dataF$marital_married <- as.factor(dataF$marital_married)
dataF$contact_cellular <- as.factor(dataF$contact_cellular)
dataF$contact_telephone <- as.factor(dataF$contact_telephone)
dataF$poutcome_failure <- as.factor(dataF$poutcome_failure)
dataF$poutcome_other <- as.factor(dataF$poutcome_other)
dataF$poutcome_success <- as.factor(dataF$poutcome_success)
library(ggplot2) 
install.packages('caret', dependencies = TRUE)
#TYPE YES! 
library(caret) 
library(corrplot) 
library(DALEX) 
library(doParallel) 
library(dplyr) 
library(inspectdf) 
library(readr)
####balancing the dataset
install.packages('ROSE')
library('ROSE')
set.seed(123)
bank <- ovun.sample(y~ .,data = dataF, method = "both",N = nrow(dataF))$data
#Smoting- may try
#install.packages('DMwR')
#install.packages('grid')
#install.packages('caTools')
#library('DMwR')
#library('grid')
#library('caTools')
#for SMOTE
#install.packages('themis')
#library(themis)
#library
#set.seed(1234)
#bank2 <- SMOTE(y~ .,data = dataF, perc.over = 40000, k = 5, perc.under = 5000)
###############training testing split-
#random split -
dt = sort(sample(nrow(dataF), nrow(dataF)*.7))
train3 <-dataF[dt,]
test3 <-dataF[-dt,]
# performing stratified sampling using caret package
require(caTools)
set.seed(101) 
sample = sample.split(bank$y, SplitRatio = .75)
banktrain = subset(bank, sample == TRUE)
banktest  = subset(bank, sample == FALSE)
banktest$y <- make.names(banktest$y)
#stratified sampling 2
trainIndex <- createDataPartition(dataF$y, p = 0.7, list = FALSE)
train2 <- dataF[ trainIndex,]
test2  <- dataF[-trainIndex,]
## we will  be using the balanced data using ROSE(bank) and the stratified sampling(train/test)
#MODELING 
summary(banktrain)
head(banktrain)
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)  
#-----GLM
print(sum(banktrain$y == 0))
set.seed(123)
glm <- train(make.names(y)~.,
             data = banktrain,
             method = "glm",
             family = "binomial",
             trControl = control)
print(glm)
#----RANDOM Forest
rfGrid <- expand.grid(mtry = 20)
set.seed(123)
rf1 <- train(make.names(y)~.,
             data = banktrain,
             method = "rf",
             ntree = 20,
             tuneLength = 5,
             trControl = control,
             tuneGrid = rfGrid)
print(rf1)
varimp <- varImp(rf1)
varimp
plot(rf1)
rf<- rf1$bestTune
rf
#Accuracy was used to select the optimal model using the largest value.
#---GBM 
set.seed(123) 
gbmGrid <- expand.grid(shrinkage = 0.1, n.trees=100,interaction.depth =4,n.minobsinnode=10)
gbm1 <- train(make.names(y)~., data =banktrain,method="gbm",trControl = control, tuneGrid = gbmGrid)
print(gbm1)
plot(gbm1)
gbm<-gbm1$bestTune
gbm
make.names(banktest$y)
#---prediction
pred_glm_raw <- predict.train(glm,newdata = banktest,type = "raw")
pred_rf_raw <- predict.train(rf1,newdata = banktest, type = "raw")
pred_gbm_raw <- predict.train(gbm1,newdata = banktest,type = "raw")
#---confusion matrix 
confusionMatrix(data = pred_glm_raw,factor(banktest$y))
confusionMatrix(data = pred_rf_raw,factor(banktest$y)) 
confusionMatrix(data = pred_gbm_raw,factor(banktest$y))
#Confusion Matrix and Statistics
#Reference
#Prediction   X0    X1
          #X0 5151  201
          #X1 496   5455

#Accuracy : 0.9383          
#95% CI : (0.9337, 0.9427)
#No Information Rate : 0.5004          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.8767          

#Mcnemar's Test P-Value : < 2.2e-16       
#                                     
#        Sensitivity : 0.9122          
#         Specificity : 0.9645          
#       Pos Pred Value : 0.9624          
#        Neg Pred Value : 0.9167          
#             Prevalence : 0.4996          
#         Detection Rate : 0.4557          
#   Detection Prevalence : 0.4735          
#      Balanced Accuracy : 0.9383          
#                                         
#       'Positive' Class : X0   
#
#---Comparison 
model_list <- list(glm = glm,gbm = gbm1,rf=rf1)
res <- resamples(model_list)
summary(res)
bwplot(res , metric = c("Accuracy"))
compare_models(gbm1,rf1)
var.imp.gbm <- varImp((gbm1))
var.imp.gbm
var.imp.glm <- varImp(glm)
var.imp.glm
var.imp.rf<- varImp(rf1)
var.imp.rf
plot(var.imp.rf, top=4)
#Overall
#balance           100.000
#age                83.528
#day                73.794
#poutcome_success1  32.344
#month              29.688
#contact_cellular1  21.967
#previous           17.710
#housing1           15.923
#campaign2          11.846
#marital_married1   10.133
#loan1               9.958
#campaign3           8.766
#pdays1              8.547
#education2          8.385
#job_technician1     8.266
#education3          8.180
#job_blue_collar1    7.964
#job_management1     7.793
#job_admin           7.277
#campaign4           6.458



