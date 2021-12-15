library(lattice)
library(tidyr)
library(Metrics)
library (Cubist)

# dbs <- c("abalone","auto","bos_housing","cali_housing","CASP","elevators","fried","machine","MV","servo")
# #dbs <- c("MV")
# #print(dbs[1])
# 
# #instantiate output dataframe
# output <- data.frame(Run=integer(), Model=character(),Dataset=character(),Train_RMSE=double(),Train_MAE=double(),Train_R2=double(),Test_RMSE=double(),Test_MAE=double(),Test_R2=double())

# 
# #30 runs
# i = 0 #placeholder
# for (i in 1:30){
#   print(i)
#   for (db in dbs){
#     #print(db)
#     
#     #read data according to indexes
#     indexes <- read.table(paste("C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/" , db , "_index.txt", sep=""))
#     indexes <- unname(as.matrix(indexes)[i, ])
#     #print(row)
#     data <- read.csv(paste("C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/", db , ".csv", sep=""))
#     data <- data[indexes, ]
#     
#     #########################################scale and separate
#     #drop na
#     data <- data[complete.cases(data), ]
#   
#     #categorical to nuemrical
#     if(db == "abalone"){
#       data$sex <- factor(data$sex)
#       data$sex <- as.numeric(data$sex)
#     }
#     if(db == "cali_housing"){
#       #drop(data$ocean_proximity)
#       data <- data[-c(10)]
#     }
#     if(db == "servo"){
#       data$motor <- factor(data$motor)
#       data$motor <- as.numeric(data$motor)
#       
#       data$screw <- factor(data$screw)
#       data$screw <- as.numeric(data$screw)
#     }
#     if(db == "MV"){
#       data$x3 <- factor(data$x3)
#       data$x3 <- as.numeric(data$x3)
#       
#       data$x7 <- factor(data$x7)
#       data$x7 <- as.numeric(data$x7)
#       
#       data$x8 <- factor(data$x8)
#       data$x8 <- as.numeric(data$x8)
#     }
#   
#     #scale
#     norm_minmax <- function(x){
#       (x- min(x)) /(max(x)-min(x))
#     }
#     data <- as.data.frame(lapply(data, norm_minmax))
#     
#     L <- length((indexes))
#     L1 = round(L * 0.8)
#     L2 = round(L * 0.9)
#     features = ncol(data)-1
#     
#     #print(summary(data))
#     
#     #########################################load indexes
#     train_indexes = indexes[1:L1]
#     valid_indexes = indexes[(L1+1):L2]
#     testt_indexes = indexes[(L2+1):L]
#     
#     train = data[train_indexes, ]
#     valid = data[valid_indexes, ]
#     testt = data[testt_indexes, ]
#     
#     #########################################split train/test and train
#     target = train$target
#     input = train[c(1:features)]
#     
#     LM = lm(target~., data=input)
#     
#     MT = cubist(x = input, y = target)
#     #print(summary(MT))
#     
#     #print("train:")
#     #print("MT: ")
#     MT_pred = predict(MT, input)
#     train_rmse <- rmse(MT_pred,target)
#     
#     LM_pred = predict(LM, input)
#     #print("LR: ")
#     #print(rmse(LM_pred,target))
#     
#     #########################################predict and evaluate
#     target = testt$target
#     input = testt[c(1:features)]
#     MT_pred = predict(MT, input)
#     
#     #print("test")
#     #print("MT: ")
#     test_rmse <- rmse(MT_pred,target)
#     
#     LM_pred = predict(LM, input)
#     #print("LR: ")
#     #print(rmse(LM_pred,target))
#     
#     #mae(pred,target)
#     
#     #print(summary(LM))
#     #print(summary(MT))
#     #print(' ')
#     
#     #########################################print to file
#     #add run's results
#     #run, model, db, train -rmse, -mae, -rsquared, test -rmse, -mae, -rsquared
#     
#     new_entry <- data.frame(i, "M5", db, train_rmse, 0, 0, test_rmse, 0, 0)
#     names(new_entry) <- names(output)
#     output <- rbind(output, new_entry)
#     
#     #print(output)
#     
#   }
# }
#write to output

write.csv(output, file = "C:/Users/Werner/Documents/GitHub/Thesis/R/results.csv", row.names = FALSE)
