import os
import sys
#sys.path.insert(0, "../../..")

from PyGasope import Evolve
from PyTree import ModelTree
from PyForest import MTForest

from mpi4py import MPI
import time

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

comm = MPI.COMM_WORLD   
size = comm.Get_size()
rank = comm.Get_rank()
#name = MPI.Get_processor_name()

path = "/home/wvandermerwe1/lustre/MTF/results/"

##############################################################################################
#functions

def LoadDB(number,chpc=True):
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
        dbs = {0:'/home/wvandermerwe1/lustre/MTF/rsc/abalone.csv',
            1:'/home/wvandermerwe1/lustre/MTF/rsc/cali_housing.csv',
            2:'/home/wvandermerwe1/lustre/MTF/rsc/bos_housing.csv',
            3:'/home/wvandermerwe1/lustre/MTF/rsc/auto.csv',
            4:'/home/wvandermerwe1/lustre/MTF/rsc/servo.csv',
            5:'/home/wvandermerwe1/lustre/MTF/rsc/machine.csv',
            6:'/home/wvandermerwe1/lustre/MTF/rsc/elevators.csv',
            7:'/home/wvandermerwe1/lustre/MTF/rsc/CASP.csv',
            8:'/home/wvandermerwe1/lustre/MTF/rsc/fried.csv',
            9:'/home/wvandermerwe1/lustre/MTF/rsc/MV.csv',
            }
    else:
        dbs = {0:'/home/werner/Desktop/Git/Thesis/PyGasope/db/abalone.csv',
            1:'/home/werner/Desktop/Git/Thesis/PyGasope/db/cali_housing.csv',
            2:'/home/werner/Desktop/Git/Thesis/PyGasope/db/bos_housing.csv',
            3:'/home/werner/Desktop/Git/Thesis/PyGasope/db/auto.csv',
            4:'/home/werner/Desktop/Git/Thesis/PyGasope/db/servo.csv',
            5:'/home/werner/Desktop/Git/Thesis/PyGasope/db/machine.csv',
            6:'/home/werner/Desktop/Git/Thesis/PyGasope/db/elevators.csv',
            7:'/home/werner/Desktop/Git/Thesis/PyGasope/db/CASP.csv',
            8:'/home/werner/Desktop/Git/Thesis/PyGasope/db/fried.csv',
            9:'/home/werner/Desktop/Git/Thesis/PyGasope/db/MV.csv',
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

    dataY = data['target'].values
    del data['target']

    Norm = preprocessing.MinMaxScaler()
    dataX = Norm.fit_transform(data.values)
    dataY = Norm.fit_transform(dataY.reshape(-1,1))
    dataY = dataY.flatten()
    
    return dataX, dataY

def write_file_deprecated(filepath,filename, db, run, scores, exec_time):
    filetitle = filename +"_{0}_db{1}.csv".format(run, db)

    # backup print to line
    print(filetitle,'({})'.format(run),'1:\t2:\t3:')
    print("Train set:\t{:.5f}\t{:.5f}\t{:.5f}".format(scores[0,0],scores[0,1],scores[0,2]))
    print("Test  set:\t{:.5f}\t{:.5f}\t{:.5f}".format(scores[1,0],scores[1,1],scores[1,2]))
    print(exec_time[0],'\t',exec_time[1],'\t',exec_time[2])
    
    f = open(os.path.join(filepath,'db{}'.format(db), filetitle), 'w')
    f.write("dataset,1,2,3")
    f.write("\n{},{},{},{}".format('Train set',scores[0,0],scores[0,1],scores[0,2]))
    f.write("\n{},{},{},{}".format('Test  set',scores[1,0],scores[1,1],scores[1,2]))
    f.write("\nExec Time,{},{},{}".format(exec_time[0],exec_time[1],exec_time[2]))
    f.flush()
    f.close()

def write_file(filepath,filename, params, db, run, scores, exec_time):
    filetitle = filename +"_{0}_db{1}.csv".format(run, db)
   
    f = open(os.path.join(filepath,'db{}'.format(db), filetitle), 'w')
    if exec_time != None:
        f.write("ParamValue,Train MSE,Test MSE, Exec Time")

        i = 0
        for param in params:
            f.write("\n{},{},{},{}".format(param,scores[0,i],scores[1,i], exec_time[i]))
            i+=1

    else:
        f.write("ParamValue,Train MSE,Test MSE")
        
        i = 0
        for param in params:
            f.write("\n{},{},{}".format(param,scores[0,i],scores[1,i]))
            i+=1

    f.flush()
    f.close()

def write_progress(filepath,param, tot, i):
    f = open(os.path.join(filepath,'PROGRESS.txt'), 'w')
    f.write("Currently computing:\nParamater {} out of {}\nOn DB{}".format(param, tot, i))
    f.flush()
    f.close()
    
temp = time.time()
#############################################################################################
# Tests to perform: 
#test_num = int(sys.argv[1])

test_num = 5

# TEST 1
# filename: linearity_{run}_db{number}
# linearity estimations (training gasope for O=[1,2,3,(4)] 30 times on each dataset)

if test_num == 1:
    for i in range(10):

        #st = time.time()

        dataX, dataY = LoadDB(i)
        N = len(dataY)

        # shuffle data
        index = np.random.permutation(N)
        dataX = dataX[index,:]
        dataY = dataY[index]

        # parameters per dataset - for small datasets, computational cost is relaxed
        terms = 10
        if i == 2:
            terms = 20
        if i == 8 or i == 9:
            terms = 13

        pop_size = 30
        if i == 6 or i ==4:
            pop_size = 50

        gens = 100
        if i == 4:
            gens = 150

        l = int(N*0.8)
        t1 = int(N*0.8)
        t2 = N
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]

        mse = np.zeros((2,3))
        exec_time = []

        for j in range(3):
            write_progress(path,j,3,i)

            st = time.time()

            # train gasope
            bestIndiv,_ = Evolve(terms, trainX, trainY, 
                                population_size=pop_size, generations=gens, max_poly_order=(j+1), 
                                Progress=False, debug=False)

            # test
            predicted_y = bestIndiv.out(trainX)
            mse[0,j] = mean_squared_error(predicted_y, trainY)
            predicted_y = bestIndiv.out(testX)
            mse[1,j] = mean_squared_error(predicted_y, testY)
            
            exec_time.append(time.time()-st)

        #time = time.time() - st

        # filepath,filename, db, run, scores
        write_file_deprecated(path,'linearity',i, (rank+1), mse, exec_time)

# TEST 2
# filename: leafs_{run}_db{number}
# min leaf samples (training gasope for N=[(5),(10),15,20,25,30] 30 times on each dataset)
if test_num == 2:
    for i in range(10):

        #st = time.time()

        dataX, dataY = LoadDB(i)
        N = len(dataY)
        F = dataX.shape[1]

        # shuffle data
        index = np.random.permutation(N)
        dataX = dataX[index,:]
        dataY = dataY[index]

        # parameters per dataset - for small datasets, computational cost is relaxed
        terms = 10
        if i == 2:
            terms = 20
        if i == 8 or i == 9:
            terms = 13

        pop_size = 30
        if i == 6 or i ==4:
            pop_size = 50

        gens = 100
        if i == 4:
            gens = 150

        # max polynomial order
        max_poly = 3
        if i == 0 or i == 1 or i == 7 or i == 8 or i == 9:
            max_poly = 2
        if i == 5:
            max_poly = 1

        l = int(N*0.8)
        t1 = int(N*0.8)
        t2 = N
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]

        mse = np.zeros((2,10))
        exec_time = []
        params = []

        # paramater min leafs from 5 - 50, increment 5
        for j in range(2,11):

            write_progress(path,j,10,i)
            data_subset_index = np.random.choice(l, j)
            params.append(j*5)
            input_data = trainX[data_subset_index,:]
            target_data = trainY[data_subset_index]

            st = time.time()

            # train gasope
            bestIndiv,_ = Evolve(terms, input_data, target_data, 
                                population_size=pop_size, generations=gens, max_poly_order=max_poly, 
                                Progress=False, debug=False)

            
            # test
            predicted_y = bestIndiv.out(trainX)
            mse[0,j-2] = mean_squared_error(predicted_y, trainY)
            predicted_y = bestIndiv.out(testX)
            mse[1,j-2] = mean_squared_error(predicted_y, testY)
            
            exec_time.append(time.time()-st)

        #time = time.time() - st

        # filepath,filename, db, run, scores
        write_file(path,'minleafs', params, i, (rank+1), mse, exec_time)

# TEST 3
# filename: depth_{run}_db{number}
# tree depth (train tree for depth=[2,3,4,5,6,7] 30 times on each dataset)
if test_num == 3:
    for i in range(10):

        #st = time.time()

        dataX, dataY = LoadDB(i)
        N = len(dataY)
        F = dataX.shape[1]

        # shuffle data
        index = np.random.permutation(N)
        dataX = dataX[index,:]
        dataY = dataY[index]

        # parameters per dataset - for small datasets, computational cost is relaxed
        terms = 10
        if i == 2:
            terms = 20
        if i == 8 or i == 9:
            terms = 13

        pop_size = 30
        if i == 6 or i ==4:
            pop_size = 50

        gens = 100
        if i == 4:
            gens = 150

        # max polynomial order
        max_poly = 3
        if i == 0 or i == 1 or i == 7 or i == 8 or i == 9:
            max_poly = 2
        if i == 5:
            max_poly = 1

        # min leafs
        min_leafs = 20
        if i == 1 or i == 2:
            min_leafs = 35
        if i == 6 or i == 9:
            min_leafs = 50

        l = int(N*0.8)
        t1 = int(N*0.8)
        t2 = N
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]

        mse = np.zeros((2,9))
        exec_time = []
        params = []

        # paramater tree depth 2-10
        for j in range(2,11):

            write_progress(path,j-1,9,i)

            params.append(j)

            st = time.time()

            # train modelt ree
            tree = ModelTree(0, trainX, trainY, max_depth=j, bagged_splits=False,
                            ga_pop_size=pop_size, ga_gens=gens, ga_terms=terms,
                            max_poly_order=max_poly, min_leaf_samples=min_leafs,
                            Progress=False)

            # test
            predicted_y = tree.predict(trainX)
            mse[0,j-2] = mean_squared_error(predicted_y, trainY)
            predicted_y = tree.predict(testX)
            mse[1,j-2] = mean_squared_error(predicted_y, testY)
            
            exec_time.append(time.time()-st)

        #time = time.time() - st

        # filepath,filename, db, run, scores
        write_file(path,'treedepth', params, i, (rank+1), mse, exec_time)

# test 4 ensemble size
if test_num == 4:
    for i in range(10):

        #st = time.time()

        dataX, dataY = LoadDB(i)
        N = len(dataY)
        F = dataX.shape[1]

        # shuffle data
        index = np.random.permutation(N)
        dataX = dataX[index,:]
        dataY = dataY[index]

        # parameters per dataset - for small datasets, computational cost is relaxed
        terms = 10
        if i == 2:
            terms = 20
        if i == 8 or i == 9:
            terms = 13

        pop_size = 30
        if i == 6 or i ==4:
            pop_size = 50

        gens = 100
        if i == 4:
            gens = 150

        # max polynomial order
        max_poly = 3
        if i == 0 or i == 1 or i == 6 or i == 7 or i == 8 or i == 9:
            max_poly = 2
        if i == 5:
            max_poly = 1

        # min leafs
        min_leafs = 20
        if i == 1 or i == 2:
            min_leafs = 35
        if i == 6 or i == 9:
            min_leafs = 50

        # tree depth
        depth = 2
        if i == 1 or i == 5 or i == 7:
            depth = 4
        if i == 6 or i == 8 or i == 9:
            depth = 5

        l = int(N*0.8)
        t1 = int(N*0.9)
        t2 = N
        validX = dataX[l:t1,:]
        validY = dataY[l:t1]
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]

        forest_size = 50
        if i == 1 or i == 6 or i == 7 or i == 8 or i == 9:
            forest_size = 30

        mse = np.zeros((2,forest_size))
        
        params = np.arange(1,forest_size)

        write_progress(path,1,1,i)

        #params.append(j)

        st = time.time()

        # train model tree forest
        MTF = MTForest(trainX, trainY, score_in=testX, score_target=testY,
                        tree_depth=depth, forest_size=forest_size, 
                        ga_pop_size=pop_size, ga_gens=gens, ga_terms=terms,
                        max_poly_order=max_poly, min_leaf_samples=min_leafs,
                        Progress=False)

        # test
        mse = MTF.score

        '''
        MTF.ForestPruning(validX, validY, printout=False)
        
        predicted_y = MTF.predict(trainX)
        mse[0,-1] = mean_squared_error(predicted_y, trainY)
        predicted_y = MTF.predict(testX)
        mse[1,-1] = mean_squared_error(predicted_y, testY)
        '''

        #exec_time.append(time.time()-st)

        end = time.time() - st

        print("execution_time on db{}: {:.2f}".format(i,end))

        # filepath,filename, db, run, scores
        write_file(path,'forestsize', params, i, (rank+1), mse, None)


# test 5 pruning
if test_num == 5:
    for i in range(10):
        
        if i != 2:
            continue

        #st = time.time()

        dataX, dataY = LoadDB(i, chpc=False)
        N = len(dataY)
        F = dataX.shape[1]

        # shuffle data
        index = np.random.permutation(N)
        dataX = dataX[index,:]
        dataY = dataY[index]
        
        l = int(N*0.8)
        t1 = int(N*0.9)
        t2 = N
        validX = dataX[l:t1,:]
        validY = dataY[l:t1]
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]

        # parameters per dataset - for small datasets, computational cost is relaxed
        terms = 10
        if i == 2:
            terms = 20
        if i == 8 or i == 9:
            terms = 13

        pop_size = 30
        if i == 6 or i ==4:
            pop_size = 50

        gens = 100
        if i == 4:
            gens = 150

        # max polynomial order
        max_poly = 3
        if i == 0 or i == 1 or i == 6 or i == 7 or i == 8 or i == 9:
            max_poly = 2
        if i == 5:
            max_poly = 1

        # min leafs
        min_leafs = 20
        if i == 1:# or i == 2:
            min_leafs = 35
        if i == 6 or i == 9:
            min_leafs = 50

        # tree depth
        depth = 2
        if i == 1 or i == 5 or i == 7:
            depth = 4
        if i == 6 or i == 8 or i == 9:
            depth = 5


        forest_size = 100
        if i == 1 or i == 6 or i == 7 or i == 8 or i == 9:
            forest_size = 30

        mse = np.zeros((2,forest_size))
        
        params = np.arange(1,forest_size)

        #write_progress(path,1,1,i)

        #params.append(j)

        st = time.time()

        # train model tree forest
        MTF = MTForest(trainX, trainY, score_in=testX, score_target=testY,
                        tree_depth=depth, forest_size=forest_size, 
                        ga_pop_size=pop_size, ga_gens=gens, ga_terms=terms,
                        max_poly_order=max_poly, min_leaf_samples=min_leafs,
                        lin_base=False, feature_bagging=False,
                        Progress=True)

        # test
        mse = MTF.score
        MTF.Print_Forest(1)

        MTF.ForestPruning(validX, validY, printout=True)

        #MTF.Forest_details()

        predicted_y = MTF.predict(trainX)
        mse[0,-1] = mean_squared_error(predicted_y, trainY)
        predicted_y = MTF.predict(testX)
        mse[1,-1] = mean_squared_error(predicted_y, testY)

        #exec_time.append(time.time()-st)

        end = time.time() - st

        #print("execution_time on db{}: {:.2f}".format(i,end))

        # filepath,filename, db, run, scores
        #write_file(path,'forestsize', params, i, (rank+1), mse, None)
        print('W/o lin base models:\n',mse)


        #################WITH LIN BASE MODELS###############################################33
        # train model tree forest
        MTF = MTForest(trainX, trainY, score_in=testX, score_target=testY,
                        tree_depth=depth, forest_size=forest_size, 
                        ga_pop_size=pop_size, ga_gens=gens, ga_terms=terms,
                        max_poly_order=max_poly, min_leaf_samples=min_leafs,
                        lin_base=True, feature_bagging=False,
                        Progress=True)

        # test
        mse = MTF.score
        MTF.Print_Forest(2)

        MTF.ForestPruning(validX, validY, printout=True)

        #MTF.Forest_details()
        
        predicted_y = MTF.predict(trainX)
        mse[0,-1] = mean_squared_error(predicted_y, trainY)
        predicted_y = MTF.predict(testX)
        mse[1,-1] = mean_squared_error(predicted_y, testY)

        #exec_time.append(time.time()-st)

        end = time.time() - st

        #print("execution_time on db{}: {:.2f}".format(i,end))

        # filepath,filename, db, run, scores
        #write_file(path,'forestsize', params, i, (rank+1), mse, None)
        print('W lin base models:\n',mse)

##############################################################################################
#print("Total single process exec time {}".format(time.time()-temp))
