from os import name
import os
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
#from pandas.io.formats import style
import seaborn as sb
import time

from pandas.core.arrays.sparse import dtype
from scipy.sparse.construct import random
from scipy.sparse.sputils import validateaxis

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from tqdm import tqdm

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

path = "C:/Users/Werner/Documents/GitHub/Thesis/Results"

import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import from_numpy as toTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss

class Network(nn.Module):
    def __init__(self, features, hidden_layer):
        super().__init__()
        self.hidden = nn.Linear(features, hidden_layer)
        self.output = nn.Linear(hidden_layer, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = F.leaky_relu(self.output(x))
    
        return x

def write_params(path, model, params, header):
    #params is an array of dictionaries equal length to the number of dbs
    #each dictionary contains the tuning results for the indexed db
    filetitle = model +"_tuning.csv"
    
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
    
    f = open(os.path.join(path,model, filetitle), 'w')

    #write header
    f.write("Dataset")
    for param in header:
        f.write(",{}".format(param))
    
    for i in range(10):
        f.write('\n{}'.format(dbs[i]))

        for value in params[i]:
            f.write(",{}".format(value))
    
    f.flush()
    f.close()
 
def LoadDB(number , chpc=False):

    if chpc:
        dbs = {0:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/abalone.csv',
            1:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/cali_housing.csv',
            2:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/bos_housing.csv',
            3:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/auto.csv',
            4:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/servo.csv',
            5:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/machine.csv',
            6:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/elevators.csv',
            7:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/CASP.csv',
            8:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/fried.csv',
            9:'/home/wvandermerwe1/lustre/MTF/rsc/dbs/db/MV.csv',
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

    dataY = data['target'].values
    del data['target']

    Norm = preprocessing.MinMaxScaler()
    dataX = Norm.fit_transform(data.values)
    dataY = Norm.fit_transform(dataY.reshape(-1,1))
    dataY = dataY.flatten()
    
    del data

    return dataX, dataY

######################################RandomForest########################################################

def tuneRF():
    params = np.zeros((10,4),dtype=int)
    for db in tqdm(range(10)):
        #Bayesian optimisations
        class tqdm_skopt(object):
            def __init__(self, **kwargs):
                self._bar = tqdm(**kwargs,leave=False)
                
            def __call__(self, res):
                self._bar.update()

        dataX, dataY = LoadDB(db)
        F = dataX.shape[1]
        N = len(dataY)

        if db >= 7:
            header = ['max_features','min_samples_leaf','max_depth','n_estimators']
            space = [Integer(2,F,name='max_features'), 
                    Integer(1,20, name='min_samples_leaf'),
                    Integer(20,150, name='max_depth'),
                    Integer(100,250, name='n_estimators')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]

        elif db == 1 or db ==6:
            header = ['max_features','min_samples_leaf','max_depth','n_estimators']
            space = [Integer(2,F,name='max_features'), 
                    Integer(1,20, name='min_samples_leaf'),
                    Integer(20,150, name='max_depth'),
                    Integer(100,500, name='n_estimators')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]

        else:
            header = ['max_features','min_samples_leaf','max_depth','n_estimators']
            space = [Integer(2,F,name='max_features'), 
                    Integer(1,20, name='min_samples_leaf'),
                    Integer(20,250, name='max_depth'),
                    Integer(100,1000, name='n_estimators')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]

        @use_named_args(space)
        def objective(**params):

            # 3 fold CV
            K = 3 #folds
            ave_mse = 0
            for i in range(K):

                #indexing
                fold_size = N/K
                fold_index = np.arange(i*fold_size, i*fold_size+fold_size, 1, dtype=int)
        
                mask = np.ones(N, bool)
                mask[fold_index] = 0
                reverse_index = np.arange(0,N,1)[mask]
                #print(reverse_index)
                
                testX = dataX[fold_index, :]
                testY = dataY[fold_index]
                trainX = dataX[reverse_index, :]
                trainY = dataY[reverse_index]

                #train and evaluate model
                rf = RandomForestRegressor(
                    n_estimators=params['n_estimators'], 
                    max_depth=params['max_depth'], 
                    max_features=params['max_features'],
                    min_samples_leaf=params['min_samples_leaf'],
                    n_jobs=-1)
            
                rf.fit(trainX, trainY)
                
                predicted = rf.predict(testX)
                mse = mean_squared_error(predicted,testY) 
                ave_mse += mse
                #print("Fold {} score:".format(i),mse)
            
            #print(ave_mse/K)
            return ave_mse/K

        n_calls = 100
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=10,
            callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")], n_jobs=-1)
        params[db,:] = res_gp.x

        #print("\nDB:",db,"\nTest Score", res_gp.fun)
        #print("Tuned parameters:",res_gp.x)
        #plot_convergence(res_gp)

        write_params(path, "RF", params, header)

#######################################SVR##########################################################

def tuneSVR():
    params = np.zeros((10,2))
    for db in tqdm(range(10)):
        
        #Bayesian optimisations
        class tqdm_skopt(object):
            def __init__(self, **kwargs):
                self._bar = tqdm(**kwargs,leave=True)
                
            def __call__(self, res):
                self._bar.update()

        dataX, dataY = LoadDB(db)
        N = len(dataY)

        header = ['C','gamma']

        space = [Integer(1,200,name='C'), 
                Real(0.01,2, prior = 'log-uniform', name='gamma')]
                #Categorical(('linear','rbf'), name = 'kernel')]
                
        @use_named_args(space)
        def objective(**params):

            # 3 fold CV
            K = 3 #folds
            ave_mse = 0
            for i in range(K):
                #indexing
                fold_size = N/K
                fold_index = np.arange(i*fold_size, i*fold_size+fold_size, 1, dtype=int)
        
                mask = np.ones(N, bool)
                mask[fold_index] = 0
                reverse_index = np.arange(0,N,1)[mask]
                #print(reverse_index)
                
                testX = dataX[fold_index, :]
                testY = dataY[fold_index]
                trainX = dataX[reverse_index, :]
                trainY = dataY[reverse_index]

                #train and evaluate model
                svm = SVR(
                    kernel='rbf', 
                    C=params['C'], 
                    gamma=params['gamma'])
            
                svm.fit(trainX, trainY)
                
                predicted = svm.predict(testX)
                mse = mean_squared_error(predicted,testY) 
                ave_mse += mse

            return ave_mse/K

        n_calls = 100
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=10,
            callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")], n_jobs=-1)
        params[db,:] = res_gp.x

        #print("\nDB:",db,"\nTest Score", res_gp.fun)
        #print("Tuned parameters:",res_gp.x)
        #plot_convergence(res_gp)
        #plot_convergence(res_gp)

        write_params(path, "SVR", params, header)

##################################neural network############################################################

def tuneNN():
    params = np.zeros((10,3))
    for db in tqdm(range(10)):
                #Bayesian optimisations
        class tqdm_skopt(object):
            def __init__(self, **kwargs):
                self._bar = tqdm(**kwargs)
                
            def __call__(self, res):
                self._bar.update()

        #data
        dataX, dataY = LoadDB(db)
        F = dataX.shape[1]
        N = len(dataY)

        MaxEpochs = 50
        if db == 2 or db == 3:
            MaxEpochs = 100
        if db == 4 or db == 5:
            MaxEpochs = 200

        if db > 5 or db == 1:


            header = ['hidden_size','batch_size','lr']
            space = [Integer(1,2*F,name='hidden_size'), 
                    Categorical([128 , 256, 512, 1024], name='batch_size'),
                    Categorical([0.003,0.001,0.0003], name='lr')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]
        
        elif db == 0:

            header = ['hidden_size','batch_size','lr']              
            space = [Integer(1,2*F,name='hidden_size'), 
                    Categorical([16, 32 ,64, 128, 256], name='batch_size'),
                    Categorical([0.003,0.001,0.0003], name='lr')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]

        else:

            header = ['hidden_size','batch_size','lr']
            space = [Integer(1,2*F,name='hidden_size'), 
                    Categorical([2, 4, 8, 16,32 ,64], name='batch_size'),
                    Categorical([0.003,0.001,0.0003], name='lr')]
                    #Real(1.03,10, prior='log-uniform',name='gamma')]

        @use_named_args(space)
        def objective(**params):

            #metric for dataframe
            train_loss=[]
            test_loss=[]
            fold=[]
            #fold2=[]
            epochs=[]
            #epochs2=[]
            mse=[]
            dataset=[]

            # 3 fold CV
            K = 3 #folds
            #ave_mse = 0
            for i in tqdm(range(K),leave=False):

                #indexing
                fold_size = N/K
                fold_index = np.arange(i*fold_size, i*fold_size+fold_size, 1, dtype=int)

                mask = np.ones(N, bool)
                mask[fold_index] = 0
                reverse_index = np.arange(0,N,1)[mask]
                #print(reverse_index)
                
                #convert to tensor
                testX = toTensor(dataX[fold_index, :]).float()
                testY = toTensor(dataY[fold_index]).float()
                trainX = toTensor(dataX[reverse_index, :]).float()
                trainY = toTensor(dataY[reverse_index]).float()

                #init
                #net = Network(F,F)
                net = Network(F, params['hidden_size'])
                #print(net.eval())

                #batch_size = 20
                batch_size = int(params['batch_size'])
                #print("\nhos kyk hier:",batch_size,type(batch_size))
                #lr = 0.001
                lr = params['lr']
                
                optim = Adam(net.parameters(), lr=lr)
                crite = MSELoss()

                #train
                for epoch in tqdm(range(MaxEpochs),leave=False):
                    
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
                    
                    #print(loss.item())
                    
                    #calc loss
                    epochs.append(epoch)
                    #epochs2.append(epoch)
                    #epochs2.append(epoch)
                    fold.append(i+1)
                    #fold2.append(i+1)
                    #fold2.append(i+1)

                    predY = net(trainX)
                    trainMSE = mean_squared_error(predY.detach().numpy(),trainY.detach().numpy())
                    train_loss.append(trainMSE)
                    mse.append(trainMSE)
                    dataset.append("Training Set")

                    predY = net(testX)
                    testMSE = mean_squared_error(predY.detach().numpy(),testY.detach().numpy())
                    test_loss.append(testMSE)
                    #mse.append(testMSE)
                    #dataset.append("Test Set")

            data = {'Fold':fold,
                'Epoch':epochs,
                'Train MSE':train_loss,
                'Test MSE':test_loss}

            '''
            data2 = {'Fold':fold2,
                'Epoch':epochs2,
                'Dataset':dataset,
                'MSE':mse}
            '''

            results = pd.DataFrame(data)
            #results2 = pd.DataFrame(data2)
            #print(results.groupby(['Epoch']).mean())

            #sb.relplot(data=results.groupby(['Epoch']).mean(), x='Epoch',y='Train MSE',kind='line')
            #sb.relplot(data=results.groupby(['Epoch']).mean(), x='Epoch',y='Test MSE',kind='line')
            #sb.relplot(data=results2, x='Epoch',y='MSE',hue='Fold',style='Dataset',kind='line')
            #print(results.groupby(['Epoch']).mean()['Test MSE'].min())
            '''
            plt.plot(train_loss)
            plt.plot(test_loss)
            plt.show()

            predY = net(testX).detach().numpy().flatten()
            testY = testY.detach().numpy()
            plt.scatter(predY,testY)
            straight_line = np.arange(0,1,0.05)
            plt.plot(straight_line,straight_line)
            '''
            #plt.show()
            return results.groupby(['Epoch']).mean()['Test MSE'].min()

        n_calls = 100
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=10,
            callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")], n_jobs=-1)
        params[db,:] = res_gp.x
        #print(params)
        #print("\nDB:",db,"\nTest Score", res_gp.fun)
        #print("Tuned parameters:",res_gp.x)
        #plot_convergence(res_gp)
        #plot_convergence(res_gp)

        write_params(path, "NN", params, header)
    #plt.show()

####################################################################################################
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


def WorstTime():
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

    for db in range(10):

        if db != 0:
            continue
        
        print(dbs[db])
        print(db)
        start = time.time()

        #data
        dataX, dataY = LoadDB(db)
        F = dataX.shape[1]
        N = len(dataY)    
        mse = 0

        #indexing
        index = int(N*0.66)

        #print(reverse_index)
        trainX = dataX[0:index, :]
        trainY = dataY[0:index]
        testX = dataX[index:N, :]
        testY = dataY[index:N]

        #train and evaluate model
        '''
        rf = RandomForestRegressor(
            n_estimators=500, 
            max_depth=200, 
            n_jobs=-1)
    
        rf.fit(trainX, trainY)
        
        predicted = rf.predict(testX)
        
        svm = SVR(
            kernel='rbf', 
            C=100, 
            gamma=0.1)
    
        svm.fit(trainX, trainY)
        
        predicted = svm.predict(testX)

        mse = mean_squared_error(predicted,testY) 
        print(time.time() - start)
        '''
        
        #metric for dataframe
        train_loss=[]
        test_loss=[]
        fold=[]
        #fold2=[]
        epochs=[]
        #epochs2=[]
        mse=[]
        dataset=[]
        
        #convert to tensor
        testX = toTensor(testX).float()
        testY = toTensor(testY).float()
        trainX = toTensor(trainX).float()
        trainY = toTensor(trainY).float()

        params = LoadParams("NN", db)
        #hidden_size,batch_size,lr

        net = Network(F, params[0])
        batch_size = params[1]
        lr = params[2]
        
        optim = Adam(net.parameters(), lr=lr)
        crite = MSELoss()

        #train
        for epoch in tqdm(range(200),leave=False):
            
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

            epochs.append(epoch)
            #epochs2.append(epoch)
            #epochs2.append(epoch)
            fold.append(1)
            #fold2.append(i+1)
            #fold2.append(i+1)

            predY = net(trainX)
            trainMSE = mean_squared_error(predY.detach().numpy(),trainY.detach().numpy())
            train_loss.append(trainMSE)
            mse.append(trainMSE)
            dataset.append("Training Set")

            predY = net(testX)
            testMSE = mean_squared_error(predY.detach().numpy(),testY.detach().numpy())
            test_loss.append(testMSE)
            
            #print(loss.item())

        data = {'Fold':fold,    
            'Epoch':epochs,
            'Train MSE':train_loss,
            'Test MSE':test_loss}

        '''
        data2 = {'Fold':fold2,
            'Epoch':epochs2,
            'Dataset':dataset,
            'MSE':mse}
        '''

        results = pd.DataFrame(data)
        #results2 = pd.DataFrame(data2)
        #print(results.groupby(['Epoch']).mean())

        sb.relplot(data=results.groupby(['Epoch']).mean(), x='Epoch',y='Train MSE',kind='line')
        sb.relplot(data=results.groupby(['Epoch']).mean(), x='Epoch',y='Test MSE',kind='line')
        #sb.relplot(data=results2, x='Epoch',y='MSE',hue='Fold',style='Dataset',kind='line')
        #print(results.groupby(['Epoch']).mean()['Test MSE'].min())
        print(time.time()-start)
        plt.show()

#tuneSVR()
#tuneRF()
#WorstTime()
#tuneNN()
tuneSVR()
