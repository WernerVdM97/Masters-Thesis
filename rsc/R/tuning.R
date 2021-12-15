library(caret)
library(doParallel)
library (Cubist)

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

dbs <- c("abalone","auto","bos_housing","cali_housing","CASP","elevators","fried","machine","MV","servo")
#dbs <- c("abalone","auto","bos_housing")

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
  if(db == "cali_housing"){
    #drop(data$ocean_proximity)
    data <- data[-c(10)]
  }
  if(db == "servo"){
    data$motor <- factor(data$motor)
    data$motor <- as.numeric(data$motor)
    
    data$screw <- factor(data$screw)
    data$screw <- as.numeric(data$screw)
  }
  if(db == "MV"){
    data$x3 <- factor(data$x3)
    data$x3 <- as.numeric(data$x3)
    
    data$x7 <- factor(data$x7)
    data$x7 <- as.numeric(data$x7)
    
    data$x8 <- factor(data$x8)
    data$x8 <- as.numeric(data$x8)
  }
  
  #scale
  norm_minmax <- function(x){
    (x- min(x)) /(max(x)-min(x))
  }
  data <- as.data.frame(lapply(data, norm_minmax))
  features = ncol(data)-1
  
  target = data$target
  
  if(db != "CASP"){
    input = data[c(1:features)]
  }else{
    input = data[c(1:features+1)]
  }
  
  #########################################tune
  grid <- expand.grid(committees = c(1, 10, 25, 50, 100), neighbors = c(0, 1, 3, 5, 7, 9))
  
  caret_grid <- train(
    x = input,
    y = target,
    method = "cubist",
    tuneGrid = grid,
    trControl = trainControl(method = "cv", number=3),
    metric="RMSE"
  )
  print(caret_grid)
}

stopCluster(cl)














