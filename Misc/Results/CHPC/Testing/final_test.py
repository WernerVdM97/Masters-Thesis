import os
import sys

from pandas.core.indexing import convert_missing_indexer

chpc = True

if not chpc:
    sys.path.append('C:/Users/Werner/Documents/GitHub/Thesis/rsc/PyGasope')

from PyForest import MTForest

if chpc:
    from mpi4py import MPI
else:
    from tqdm import tqdm 

import time

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

if not chpc:
    import torch
    from torch import from_numpy as toTensor
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.nn import MSELoss

if chpc:
    comm = MPI.COMM_WORLD   
    size = comm.Get_size()
    rank = comm.Get_rank() #instance number out of size
    #rank = 1
    #name = MPI.Get_processor_name()

    path = "/home/wvandermerwe1/lustre/Testing/results/"
else:
    path = "C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests"

def LoadParams(model, db):    
    '''
    0 - abalone
    1 - cali housing
    2 - bos housing
    3 - auto mpg
    4 - servo
    5 - machine
    6 - elevators
    7 - CASP
    8 - Friedman Artificial dataset
    9 - Luis Torgo Artificial dataset
    '''
    if model == "NN":
        #hidden_size,batch_size,lr
        params={
            0: [11, 16, 0.003],
            1: [12, 128, 0.003],
            2: [8, 2, 0.001],
            3: [12, 32, 0.003],
            4: [8, 32, 0.003],
            5: [3, 2, 0.001],
            6: [36, 128, 0.003],
            7: [18, 128, 0.003],
            8: [20, 256, 0.003],
            9: [18, 128, 0.003]
            }
        return params[db]    
    
    elif model == "SVR":
        #C, gamma
        params={
            0: [20, 0.6162542902218026],
            1: [7, 2.0],
            2: [21, 0.02439988708316495],
            3: [1, 0.42299644001587744],
            4: [1, 0.888701771283583],
            5: [8, 0.04232577718978212],
            6: [200, 0.11375408824570461],
            7: [200, 2.0],
            8: [200, 0.11342269896760986],
            9: [1, 1.0836998282267452]
            }
        return params[db]

    elif model == "RF":
        #max_features,min_samples_leaf,max_depth,n_estimators
        params={
            0: [5, 10, 28, 122],
            1: [3, 1, 150, 500],
            2: [6, 1, 20, 1000],
            3: [2, 1, 250, 1000],
            4: [3, 1, 250, 1000],
            5: [2, 1, 250, 1000],
            6: [16, 1, 145, 444],
            7: [5, 1, 150, 250],
            8: [10, 2, 146, 250],
            9: [8, 1, 76, 239]
            }
        return params[db]    
    
    elif model == "MTF":
        #terms, popsize, gens, maxpoly, minleafs, depth, forests size
        params={
            0: [10, 30, 100, 2, 20, 3, 20],
            1: [10, 30, 100, 2, 20, 4, 25],
            2: [20, 30, 100, 3, 20, 2, 30],
            3: [10, 30, 100, 3, 20, 2, 30],
            4: [10, 50, 150, 3, 20, 2, 30],
            5: [10, 30, 100, 1, 20, 4, 25],
            6: [10, 50, 100, 2, 20, 6, 25],
            7: [10, 30, 100, 2, 20, 2, 25],
            8: [13, 30, 100, 2, 20, 6, 20],
            9: [13, 30, 100, 2, 20, 6, 20]
            }
        return params[db]

if not chpc:
    class Network(nn.Module):
        def __init__(self, features, hidden_layer):
            super().__init__()
            self.hidden = nn.Linear(features, hidden_layer)
            self.output = nn.Linear(hidden_layer, 1)
        
        def forward(self, x):
            x = F.leaky_relu(self.hidden(x))
            x = F.leaky_relu(self.output(x))
        
            return x

#Load the database according to the shuffled indexes
def LoadDB(number, run=None):
    '''
    0 - abalone
    1 - cali housing
    2 - bos housing
    3 - auto mpg
    4 - servo
    5 - machine
    6 - elevators
    7 - CASP
    8 - Friedman Artificial dataset
    9 - Luis Torgo Artificial dataset
    '''
    if chpc:
        dbs = {
            0:'/home/wvandermerwe1/lustre/Testing/dbs/db/abalone.csv',
            1:'/home/wvandermerwe1/lustre/Testing/dbs/db/cali_housing.csv',
            2:'/home/wvandermerwe1/lustre/Testing/dbs/db/bos_housing.csv',
            3:'/home/wvandermerwe1/lustre/Testing/dbs/db/auto.csv',
            4:'/home/wvandermerwe1/lustre/Testing/dbs/db/servo.csv',
            5:'/home/wvandermerwe1/lustre/Testing/dbs/db/machine.csv',
            6:'/home/wvandermerwe1/lustre/Testing/dbs/db/elevators.csv',
            7:'/home/wvandermerwe1/lustre/Testing/dbs/db/CASP.csv',
            8:'/home/wvandermerwe1/lustre/Testing/dbs/db/fried.csv',
            9:'/home/wvandermerwe1/lustre/Testing/dbs/db/MV.csv',
            }
    else:
        dbs = {0:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/abalone.csv',
            1:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/cali_housing.csv',
            2:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/bos_housing.csv',
            3:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/auto.csv',
            4:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/servo.csv',
            5:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/machine.csv',
            6:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/elevators.csv',
            7:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/CASP.csv',
            8:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/fried.csv',
            9:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/MV.csv',
            }

    if chpc:
        dbs_index = {
            0:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/abalone_index.txt',
            1:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/cali_housing_index.txt',
            2:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/bos_housing_index.txt',
            3:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/auto_index.txt',
            4:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/servo_index.txt',
            5:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/machine_index.txt',
            6:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/elevators_index.txt',
            7:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/CASP_index.txt',
            8:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/fried_index.txt',
            9:'/home/wvandermerwe1/lustre/Testing/dbs/db_index/MV_index.txt',
            }
    else:
        dbs_index = {0:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/abalone_index.txt',
            1:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/cali_housing_index.txt',
            2:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/bos_housing_index.txt',
            3:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/auto_index.txt',
            4:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/servo_index.txt',
            5:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/machine_index.txt',
            6:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/elevators_index.txt',
            7:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/CASP_index.txt',
            8:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/fried_index.txt',
            9:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/MV_index.txt',
            }

    data = pd.read_csv(dbs[number])
    data = data.dropna()

    if number == 0:
        data['sex'] = data['sex'].astype('category')

        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

    if number == 1:
        del data['ocean_proximity']

    if number == 4:
        data['motor'] = data['motor'].astype('category')
        data['screw'] = data['screw'].astype('category')

        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

    if number == 9:
        data['x3'] = data['x3'].astype('category')
        data['x7'] = data['x7'].astype('category')
        data['x8'] = data['x8'].astype('category')

        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)        

    #print(data.head())

    # reorder according to indexes
    indexes = np.loadtxt(dbs_index[number])

    if not chpc:
        data = data.reindex(indexes[run,:])
    else:
        data = data.reindex(indexes[rank,:])
    
    dataY = data['target'].values
    del data['target']

    Norm = preprocessing.MinMaxScaler()
    dataX = Norm.fit_transform(data.values)
    dataY = Norm.fit_transform(dataY.reshape(-1,1))
    dataY = dataY.flatten()
    
    del data
    del indexes

    return dataX, dataY

#write output of tests to an txt file
def write_file(filepath, filename, model, scores):
    #assumes trains scores of the shape [30 x 10 x 2 x 3] 
    #i.e. [run #] by [db 0-9] by [train, test] by [RMSE, MAE, Rsquared]
    
    if chpc:
        filetitle = filename +"_run_{0}.csv".format(rank)
    else:
        filetitle = filename +".csv"
    
    dbs = {    
        0 : 'abalone',
        1 : 'cali_housing',
        2 : 'bos_housing',
        3 : 'auto',
        4 : 'servo',
        5 : 'machine',
        6 : 'elevators',
        7 : 'CASP',
        8 : 'fried',
        9 : 'MV'}
    
    f = open(os.path.join(filepath, filetitle), 'w')
    #f = open(filetitle, 'w')

    f.write("Run,Model,Dataset,Train_RMSE,Train_MAE,Train_R2,Test_RMSE,Test_MAE,Test_R2")
    
    if not chpc:
        for i in range(30):
            for key, value in dbs.items():
                f.write("\n{},{},{},{},{},{},{},{},{}".format(i+1,
                                                            model,
                                                            value,
                                                            scores[i, key,0,0],
                                                            scores[i, key,0,1],
                                                            scores[i, key,0,2],
                                                            scores[i, key,1,0],
                                                            scores[i, key,1,1],
                                                            scores[i, key,1,2]))
    else:                                                    
        for key, value in dbs.items():
            f.write("\n{},{},{},{},{},{},{},{},{}".format(rank,
                                                        model,
                                                        value,
                                                        scores[key,0,0],
                                                        scores[key,0,1],
                                                        scores[key,0,2],
                                                        scores[key,1,0],
                                                        scores[key,1,1],
                                                        scores[key,1,2]))
        
    f.flush()
    f.close()

def TestNN():
    if not chpc:
        scores = np.zeros((30,10,2,3))
    else:
        scores = np.zeros((10,2,3))

    
    for run in range(30):

        if chpc and run > 0:
            continue

        for db in range(10):
                    
            #data
            dataX, dataY = LoadDB(db, run)

            F = dataX.shape[1]
            N = len(dataY)
            MaxEpochs = 200

            t1 = int(N*0.8)
            t2 = int(N*0.9)
            
            #convert to tensor
            testX = toTensor(dataX[t2:N, :]).float()
            testY = toTensor(dataY[t2:N]).float()
            validX = toTensor(dataX[t1:t2, :]).float()
            validY = toTensor(dataY[t1:t2]).float()
            trainX = toTensor(dataX[0:t1, :]).float()
            trainY = toTensor(dataY[0:t1]).float()

            params = LoadParams("NN", db)
            #hidden_size,batch_size,lr

            net = Network(F, params[0])
            batch_size = params[1]
            lr = params[2]
            
            optim = Adam(net.parameters(), lr=lr)
            crite = MSELoss()

            best_net = net
            best_mse = 999999999999
            
            ##########################################################################################
            #train
            for _ in range(MaxEpochs):
                
                #shuffle per epoch
                shuffled_indices = np.arange(start=0,stop=trainY.size()[0],step=1)
                np.random.shuffle(shuffled_indices)
                trainX = trainX[shuffled_indices,:]
                trainY = trainY[shuffled_indices]

                for batch, target in zip(torch.split(trainX, batch_size), torch.split(trainY, batch_size)):
                    #reset grad
                    optim.zero_grad()

                    #forward prop
                    output = net(batch)

                    #calc loss
                    target = target.reshape(target.size()[0], 1)
                    loss = crite(output, target)
                    
                    #backward prop
                    loss.backward()

                    #optim weights
                    optim.step()
                
                # save best net
                predY = net(validX)
                validMSE = mean_squared_error(predY.detach().numpy(),validY.detach().numpy())
                if validMSE < best_mse:
                    best_mse = validMSE
                    best_net = net
                
            ##########################################################################################
            #predict
            pred_trainY = best_net(trainX)
            pred_testY = best_net(testX)

            #assumes trains scores of the shape [10 x 2 x 3] 
            #i.e. [db 0-9] by [train, test] by [RMSE, MAE, Rsquared]
            if chpc:
                scores[db, 0, 0] = mean_squared_error(pred_trainY.detach().numpy(),trainY.detach().numpy())
                scores[db, 0, 1] = mean_absolute_error(pred_trainY.detach().numpy(),trainY.detach().numpy())
                scores[db, 0, 2] = r2_score(pred_trainY.detach().numpy(),trainY.detach().numpy())

                scores[db, 1, 0] = mean_squared_error(pred_testY.detach().numpy(),testY.detach().numpy())
                scores[db, 1, 1] = mean_absolute_error(pred_testY.detach().numpy(),testY.detach().numpy())
                scores[db, 1, 2] = r2_score(pred_testY.detach().numpy(),testY.detach().numpy())

            else:
                scores[run, db, 0, 0] = mean_squared_error(pred_trainY.detach().numpy(),trainY.detach().numpy())
                scores[run, db, 0, 1] = mean_absolute_error(pred_trainY.detach().numpy(),trainY.detach().numpy())
                scores[run, db, 0, 2] = r2_score(pred_trainY.detach().numpy(),trainY.detach().numpy())

                scores[run, db, 1, 0] = mean_squared_error(pred_testY.detach().numpy(),testY.detach().numpy())
                scores[run, db, 1, 1] = mean_absolute_error(pred_testY.detach().numpy(),testY.detach().numpy())
                scores[run, db, 1, 2] = r2_score(pred_testY.detach().numpy(),testY.detach().numpy())

            write_file(path, "NN_results", "NN", scores)

def TestMTF():
    if not chpc:
        scores = np.zeros((30,10,2,3))
    else:
        scores = np.zeros((10,2,3))

    for run in range(30):

        if chpc and run > 0:
            continue

        for db in range(10):

            #data
            dataX, dataY = LoadDB(db, run)

            F = dataX.shape[1]
            N = len(dataY)

            t1 = int(N*0.8)
            t2 = int(N*0.9)
            
            #convert to tensor
            testX = dataX[t2:N, :]
            testY = dataY[t2:N]
            validX = dataX[t1:t2, :]
            validY = dataY[t1:t2]
            trainX = dataX[0:t1, :]
            trainY = dataY[0:t1]

            params = LoadParams("MTF", db)
            #0terms, 1popsize, 2gens, 3maxpoly, 4minleafs, 5depth, 6forests size
            
            ##########################################################################################
            #train
            MTF = MTForest(trainX, trainY, tree_depth=params[5], forest_size=params[6], 
                            ga_pop_size=params[1], ga_gens=params[2], ga_terms=params[0],
                            max_poly_order=params[3], min_leaf_samples=params[4],
                            lin_base=True, feature_bagging=False,
                            Progress=True)
            
            MTF.ForestPruning(validX, validY)
                
            ##########################################################################################
            #predict
            pred_trainY = MTF.predict(trainX)
            pred_testY = MTF.predict(testX)
            if not chpc:
                scores[run, db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[run, db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[run, db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[run, db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[run, db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[run, db, 1, 2] = r2_score(pred_testY,testY)
            else:

                scores[db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[db, 1, 2] = r2_score(pred_testY,testY)

            write_file(path, "MTF_results", "MTF", scores)

def TestSVR():
    if not chpc:
        scores = np.zeros((30,10,2,3))
    else: 
        scores = np.zeros((10,2,3))

    for run in range(30):

        if chpc and run > 0:
            continue

        for db in range(10):

            #data
            dataX, dataY = LoadDB(db, run)
            F = dataX.shape[1]
            N = len(dataY)

            t1 = int(N*0.8)
            t2 = int(N*0.9)
            
            #convert to tensor
            testX = dataX[t2:N, :]
            testY = dataY[t2:N]
            validX = dataX[t1:t2, :]
            validY = dataY[t1:t2]
            trainX = dataX[0:t1, :]
            trainY = dataY[0:t1]

            params = LoadParams("SVR", db)
            #0gamma, 1C
            
            ##########################################################################################
            #train
            svm = SVR(  kernel='rbf', 
                        C=params[1], 
                        gamma=params[0])

            svm.fit(trainX, trainY)
                        
            ##########################################################################################
            #predict
            pred_trainY = svm.predict(trainX)
            pred_testY = svm.predict(testX)

            if chpc:
            
                scores[db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[db, 1, 2] = r2_score(pred_testY,testY)

            else:
                scores[run, db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[run, db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[run, db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[run, db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[run, db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[run, db, 1, 2] = r2_score(pred_testY,testY)

            write_file(path, "SVR_results", "SVR", scores)

def TestRF():
    if not chpc:
        scores = np.zeros((30,10,2,3))
    else:
        scores = np.zeros((10,2,3))

    for run in range(30):

        if chpc and run > 0:
            continue

        for db in range(10):

            #data
            dataX, dataY = LoadDB(db, run)
            F = dataX.shape[1]
            N = len(dataY)

            t1 = int(N*0.8)
            t2 = int(N*0.9)
            
            #convert to tensor
            testX = dataX[t2:N, :]
            testY = dataY[t2:N]
            validX = dataX[t1:t2, :]
            validY = dataY[t1:t2]
            trainX = dataX[0:t1, :]
            trainY = dataY[0:t1]

            params = LoadParams("RF", db)
            #0max_features, 1min_samples_leaf, 2max_depth, 3n_estimators               

            ##########################################################################################
            #train
            if chpc:
                rf = RandomForestRegressor(
                    n_estimators=params[3], 
                    max_depth=params[2], 
                    max_features=params[0],
                    min_samples_leaf=params[1])
            
            else:
                rf = RandomForestRegressor(
                    n_estimators=params[3], 
                    max_depth=params[2], 
                    max_features=params[0],
                    min_samples_leaf=params[1],
                    n_jobs=-1)
            
            rf.fit(trainX, trainY)
                        
            ##########################################################################################
            #predict
            pred_trainY = rf.predict(trainX)
            pred_testY = rf.predict(testX)
            
            if chpc:
                scores[db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[db, 1, 2] = r2_score(pred_testY,testY)

            else:
                scores[run, db, 0, 0] = mean_squared_error(pred_trainY,trainY)
                scores[run, db, 0, 1] = mean_absolute_error(pred_trainY,trainY)
                scores[run, db, 0, 2] = r2_score(pred_trainY,trainY)

                scores[run, db, 1, 0] = mean_squared_error(pred_testY,testY)
                scores[run, db, 1, 1] = mean_absolute_error(pred_testY,testY)
                scores[run, db, 1, 2] = r2_score(pred_testY,testY)

            write_file(path, "RF_results", "RF", scores)

#TestNN()
#TestSVR()
#TestRF()
TestMTF()
