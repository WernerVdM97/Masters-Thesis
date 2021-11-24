library (Cubist)
library(Metrics)
library(doParallel)

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

dbs <- c("CASP")

output <- data.frame(Dataset=character(),Neigbhours=integer(),Ensemble=integer())

for (db in dbs){
  data <- read.csv(paste("C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/", db , ".csv", sep=""))
  
  #########################################scale and separate
  #drop na
  data <- data[complete.cases(data), ]
  
  #categorical to nuemrical
  if(db == "abalone"){
    data$sex <- factor(data$sex)
    data$sex <- as.numeric(data$sex)
  }
  #scale
  norm_minmax <- function(x){
    (x- min(x)) /(max(x)-min(x))
  }
  data <- as.data.frame(lapply(data, norm_minmax))
  features = ncol(data)-1
  
  L <- length((data$target))
  L1 = round(L * 0.8)
  L2 = round(L * 0.9)
  
  #########################################tune
  indexes <- 1:L
  train_indexes = indexes[1:L1]
  valid_indexes = indexes[(L1+1):L2]
  testt_indexes = indexes[(L2+1):L]
  
  train = data[train_indexes, ]
  valid = data[valid_indexes, ]
  testt = data[testt_indexes, ]
  
  #train score
  target = train$target
  input = train[c(1:features+1)]
  
  MT = cubist(x = input, y = target)
  
  MT_pred = predict(MT, input)
  train_rmse <- rmse(MT_pred,target)
  
  #test score
  target = testt$target
  input = testt[c(1:features+1)]
  
  MT_pred = predict(MT, input)
  test_rmse <- rmse(MT_pred,target)
  
  grid <- expand.grid(committees = c(1, 10, 50, 100), neighbors = c(0, 1, 3, 5, 7, 9))
  
  caret_grid <- train(
    x = input,
    y = target,
    method = "cubist",
    tuneGrid = grid,
    trControl = trainControl(method = "cv", number=5),
    metric="RMSE"
  )
  print(caret_grid)
}

stopCluster(cl)
