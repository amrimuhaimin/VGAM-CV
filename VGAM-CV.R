# Load data
train <- read.csv("abalone_training.csv")
test <- read.csv("abalone_testing.csv")

# Load Library 
library(VGAM)
library(splitTools)
library(caret)
library(mldr)
library(mlr)

# Modeling
index <- as.integer(rownames(train))
folds <- create_folds(index, k=5)
acc_train <- 0
acc_test <- 0
j <- 1
for (i in 1:length(folds)){
  cv_train <- train[folds[[i]],]
  cv_test <- train[-folds[[i]],]
  model <- vgam(cbind(Gender, Rings.Class)~. + s(Length), binomialff(multiple.responses = T), data=cv_train)
  fitted <- data.frame(model@fitted.values)
  fitted_testing <- data.frame(predict(model, cv_test, type='res'))
  fitted$Gender <- ifelse(fitted$Gender > 0.5, 1, 0)
  fitted$Rings.Class <- ifelse(fitted$Rings.Class > 0.5, 1, 0)
  fitted_testing$Gender <- ifelse(fitted_testing$Gender > 0.5, 1, 0)
  fitted_testing$Rings.Class <- ifelse(fitted_testing$Rings.Class > 0.5, 1, 0)
  acc_score_training <- c(confusionMatrix(table(fitted$Rings.Class, cv_train$Rings.Class))$overall[[1]],
                         confusionMatrix(table(fitted$Gender, cv_train$Gender))$overall[[1]])
  acc_score_testing <- c(confusionMatrix(table(fitted_testing$Rings.Class, cv_test$Rings.Class))$overall[[1]],
                 confusionMatrix(table(fitted_testing$Gender, cv_test$Gender))$overall[[1]])
  acc_train[j] <- mean(acc_score_training)
  acc_test[j] <- mean(acc_score_testing)
  j <- j+1
}
acc_train
acc_test

# Choosen fold
model_all <- vgam(cbind(Gender, Rings.Class)~., binomialff(multiple.responses = T), data=train)
predict_test <- data.frame(predict(model_all, test, type='res'))
predict_test$Gender <- ifelse(predict_test$Gender > 0.5, 1, 0)
predict_test$Rings.Class <- ifelse(predict_test$Rings.Class > 0.5, 1, 0)
acc_final <- c(confusionMatrix(table(predict_test$Rings.Class, test$Rings.Class))$overall[[1]],
                        confusionMatrix(table(predict_test$Gender, test$Gender))$overall[[1]])
acc_final <- mean(acc_score_training)


# Evaluate model
test_mldr <- mldr_from_dataframe(test, labelIndices = c(8,9), name="testmldr")
result_evaluate <- mldr_evaluate(test_mldr, as.matrix(predict_test))
str(result_evaluate)

# Benchmark Multilabel Random Forest
train2 <- rbind(train,test)
train2$Gender <- ifelse(train2$Gender == 1, TRUE, FALSE)
train2$Rings.Class <- ifelse(train2$Rings.Class == 1, TRUE, FALSE)
scene.task <- makeMultilabelTask(data = train2, target = c("Gender", "Rings.Class"))
binary.learner <- makeLearner("classif.rpart")
lrncc <- makeMultilabelClassifierChainsWrapper(binary.learner)
n <- getTaskSize(scene.task)
train.set <- seq(1, 2268, by=1)
test.set <- seq(2269, 2835, by=1)
scene.mod.cc <- train(lrncc, scene.task, subset=train.set)
scene.pred.cc <- predict(scene.mod.cc, task=scene.task, subset=test.set)
mlr_fitted <- data.frame(scene.pred.cc$data$response.Gender, scene.pred.cc$data$response.Rings.Class)
colnames(mlr_fitted) <- c("Gender", "Rings.Class")
mlr_fitted$Gender <- ifelse(mlr_fitted$Gender == T, 1, 0)
mlr_fitted$Rings.Class <- ifelse(mlr_fitted$Rings.Class==T, 1, 0)
result_mlr <- mldr_evaluate(test_mldr, as.matrix(mlr_fitted))
str(result_mlr)

