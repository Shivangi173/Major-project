library(caret)
library (glmnet)
library (psych)
library(mlbench)
library(e1071)
#data
data<-read.csv(
  "C:/Users/91885/Downloads/data.csv"
)


data$diagnosis<-factor(data$diagnosis,levels = c("B","M"),labels = c(0, 1))
str(data)
?data
#normalize
str(data)
norm<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}
data_n<-as.data.frame(lapply(data[2:31],norm))


#data partition
set.seed(222)
ind<-sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train<-data[ind==1,]
test<-data[ind==2,]

#custom control parameters
custom <- trainControl(method = "repeatedcv", 
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

# Linear Model
set.seed(1234)
lm <- train(diagnosis~.,
            train,
            method = 'lm',
            trControl = custom
)

# Results
lm$results



# Ridge Regression
set.seed(1234)
ridge <- train(diagnosis~.,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length = 5)),
               
               trControl = custom)

# Plot Results
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = 'dev', label=T)
plot(varImp(ridge, scale=T))

#Lasso regression
set.seed(1234)
lasso <- train(diagnosis~., 
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=1, 
                                      lambda = seq(0.0001, 1, length = 5)),
               trControl = custom)
#result
lasso$results

# Plot Results
plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label=T)

lassoImp = (varImp(lasso, scale = TRUE))
plot(lassoImp, top = 10)

# Elastic Net Regression
set.seed(1234)
en <- train(diagnosis~., 
            train,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha=seq(0, 1, length=10), 
                                   lambda = seq(0.0001, 1, length = 5)),
            trControl = custom)

# Plot Results
plot(en)
plot(en$finalModel, xvar = 'lambda', label=T)
plot(en$finalModel, xvar = 'dev', label=T)
plot(varImp(en))

enImp = (varImp(en, scale = TRUE))
plot(enImp, top = 10)

# Compare Models
model_list <- list(  Lasso = lasso, ElasticNet = en)
res <- resamples(model_list)

summary(res)
bwplot(res)
#xyplot(res, metric = 'RMSE')

#best model
en$bestTune
best <- en$finalModel
coef(best, s = en$bestTune$lambda)

# Save Final Model for Later Use
saveRDS(en, "final_models.rds")
fms <- readRDS("final_models.rds")
print(fms)

# Prediction
p1 <- predict(fms, train)
sqrt(mean((train$diagnosis)^2))

p2 <- predict(fm, test)
sqrt(mean((test$medv-p2)^2))
