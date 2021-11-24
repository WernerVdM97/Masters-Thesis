'''
class containing examples for PyGasope with visuals
'''

from typing import Container
from numpy.testing._private.utils import jiffies
from sklearn.metrics import mean_squared_error
import sys

import math
import time
from tqdm import tqdm
#import mse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd

from ArtificialData import *
from PyGasope import Evolve
from PyForest import MTForest

#bayesian optimisation
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

#statistical tests
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy.stats import rankdata
#import Orange

from sklearn import linear_model
from sklearn.svm import SVR

def runExamples2D():
    l = 50

    data2D,z = TwoDNoise(l)
    bestIndiv,_ = Evolve(30, data2D, z, population_size=30, generations=100, max_poly_order=5, Progress=False, debug=True)

    x = np.linspace(-1, 1,l) 
    y = np.linspace(0, 0.5,l)
    predicted = bestIndiv.out(np.array([x,y]))

    print(bestIndiv)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection = "3d")
    ax.scatter3D(x,y, zs = z)
    ax.plot3D(x,y, zs = predicted)

    plt.show()

def runExamples1D():
    fig2 = plt.figure()

    ##############################################################################################
    dataY = CreateData(1,100)
    dataX = np.linspace(0,2*math.pi, 100)

    bestIndiv,_ = Evolve(10, dataX, dataY, 
                       population_size=30, generations=100, max_poly_order=5, 
                       Progress=True, debug=False)
    predicted = bestIndiv.out(dataX)

    ax1 = fig2.add_subplot(231)
    ax1.set_title("sin(x)\n"+bestIndiv.getPlotTitle())
    ax1.scatter(dataX,dataY)
    ax1.plot(dataX,predicted)

    ##############################################################################################
    dataY = CreateData2(1,100)
    dataX = np.linspace(-1.5,1.5,100)

    bestIndiv,_ = Evolve(10, dataX, dataY, population_size=30, generations=100, 
                         max_poly_order=5, Progress=False, debug=False)
    predicted = bestIndiv.out(dataX)

    ax2 = fig2.add_subplot(232)
    ax2.set_title("e^(x)\n"+bestIndiv.getPlotTitle())
    ax2.scatter(dataX,dataY)
    ax2.plot(dataX,predicted)
    ##############################################################################################

    dataY = CreateData3(1,100)
    dataX = np.linspace(0.2,2.2,100)

    bestIndiv,_ = Evolve(10, dataX, dataY, population_size=30, generations=100, 
                         max_poly_order=5, Progress=False, debug=False)
    predicted = bestIndiv.out(dataX)

    ax3 = fig2.add_subplot(233)
    ax3.set_title("1/x\n"+bestIndiv.getPlotTitle())
    ax3.scatter(dataX,dataY)
    ax3.plot(dataX,predicted)
    ##############################################################################################

    dataY = CreateData4(1,100)
    dataX = np.linspace(0.03,3.03,100)

    bestIndiv,_ = Evolve(10, dataX, dataY, population_size=30, generations=100, 
                         max_poly_order=5, Progress=False, debug=False)
    predicted = bestIndiv.out(dataX)

    ax4 = fig2.add_subplot(234)
    ax4.set_title("ln(x)\n"+bestIndiv.getPlotTitle())
    ax4.scatter(dataX,dataY)
    ax4.plot(dataX,predicted)
    ##############################################################################################

    dataY = CreateData5(1,100)
    dataX = np.linspace(-3,3,100)

    bestIndiv,_ = Evolve(10, dataX, dataY, population_size=30, generations=100, 
                         max_poly_order=5, Progress=False, debug=False)
    predicted = bestIndiv.out(dataX)

    ax5 = fig2.add_subplot(235)
    ax5.set_title("square root(x)\n"+bestIndiv.getPlotTitle())
    ax5.scatter(dataX,dataY)
    ax5.plot(dataX,predicted)
    ##############################################################################################

    dataY = CreateData6(1,100)
    dataX = np.linspace(0,3,100)

    bestIndiv,_ = Evolve(10, dataX, dataY, population_size=30, generations=100, 
                         max_poly_order=5, Progress=False, debug=False)
    predicted = bestIndiv.out(dataX)

    ax6 = fig2.add_subplot(236)
    ax6.set_title("1/(1+e^(x)) i.e. sigmoid\n"+bestIndiv.getPlotTitle())
    ax6.scatter(dataX,dataY)
    ax6.plot(dataX,predicted)

    plt.tight_layout()
    plt.show()

def runHousing():
    dataX, dataY = LoadHousing()
    #print(dataX.shape)
    #fig, ax = plt.subplots(2,2,figsize=(10,10))

    l = int(20000*0.6)
    t1 = int(20000*0.8)
    t2 = 20000
    validX = dataX[:,l:t1].T
    validY = dataY[l:t1]
    testX = dataX[:,t1:t2].T
    testY = dataY[t1:t2]
    trainX = dataX[:,:l].T
    trainY = dataY[:l]

    #################################################################
    #svr
    
    svm = SVR(kernel = 'rbf' , C = 50, gamma = 'auto')
    svm.fit(trainX, trainY)
    
    predicted_y = svm.predict(trainX)
    rmse1 = mean_squared_error(predicted_y, trainY)
    '''
    sorted_predictions, sorted_targets = SortByOut(predicted_y,trainY)
    ax[0,0].plot(sorted_predictions, 'r', label='predicted')
    ax[0,0].plot(sorted_targets, 'b', label='target')
    ax[0,0].legend()
    ax[0,0].set_title("Training set")
    ax[0,0].set_ylabel("SVR")
    '''
    
    predicted_y = svm.predict(testX)
    rmse2 = mean_squared_error(predicted_y, testY)
    '''
    sorted_predictions, sorted_targets = SortByOut(predicted_y,testY)
    ax[0,1].plot(sorted_predictions, 'r', label='predicted')
    ax[0,1].plot(sorted_targets, 'b', label='target')
    ax[0,1].legend()
    ax[0,1].set_title("Test set")
    '''
    print("SVR:\nRMSE train: {:.4f}\nRMSE test: {:.4f}\n".format(rmse1,rmse2))   


    ##################################################################
    #GASOPE
    '''
    for i in range(1,4):
        p = i
        bestIndiv,_ = Evolve(10, trainX, trainY, 
                            population_size=30, generations=100, max_poly_order=p, 
                            Progress=True, debug=False)
        
        predicted_y = bestIndiv.out(trainX)
        rmse1 = mean_squared_error(predicted_y, trainY)
        sorted_predictions, sorted_targets = SortByOut(predicted_y,trainY)
        ax[1,0].plot(sorted_predictions, 'r', label='predicted')
        ax[1,0].plot(sorted_targets, 'b', label='target')
        ax[1,0].legend()
        ax[1,0].set_title("Training set")
        ax[1,0].set_ylabel("GASOPE")

        predicted_y = bestIndiv.out(testX)
        rmse2 = mean_squared_error(predicted_y, testY)
        sorted_predictions, sorted_targets = SortByOut(predicted_y,testY)
        ax[1,1].plot(sorted_predictions, 'r', label='predicted')
        ax[1,1].plot(sorted_targets, 'b', label='target')
        ax[1,1].legend()
        ax[1,1].set_title("Test set")
        print("\nGASOPE w p={}:\nRMSE train: {:.4f}\nRMSE test: {:.4f}\n\n".format(p,rmse1,rmse2))  
        
    #plt.tight_layout()
    #plt.show()
    '''
    ##################################################################
    #forest 
    forest = MTForest(trainX,trainY, forest_size=50, max_poly_order = 2, tree_depth = 5)
    #forest.Print_Forest()
    #forest.Forest_details()
    #Analyse(dataIn,dataTarget,bestIndiv)
    
    #forest.Print_Forest()
    for _ in range(0):
        predicted = forest.predict(testX)
        rmse = CalcRMSE(predicted,testY)
        print("Root mean squared error: {:.4f}".format(rmse))
        forest.ForestPruning(validX,validY)
    #forest.Forest_details()

    predicted_y = forest.predict(trainX)
    rmse1 = mean_squared_error(predicted_y, trainY)
    predicted_y = forest.predict(testX)
    rmse2 = mean_squared_error(predicted_y, testY)

    print("\nMTF:\nRMSE train: {:.4f}\nRMSE test: {:.4f}\n\n".format(rmse1,rmse2))  
    ###################################################################
    #plot results
    '''
    sorted_predictions, sorted_targets = SortByOut(predicted,dataY)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    ax.plot(sorted_predictions, label='Predicted')
    ax.plot(sorted_targets, label='Target')
    ax.legend()
    ax.set(ylabel="Cost", xlabel="Instance #")

    plt.show()
    '''

def runAbalone():
    
    dataX, dataY = LoadAbalone()
    #seperate trianing and test sets

    l = int(4149*0.6)
    t1 = int(4149*0.8)
    t2 = 4149
    
    dataVX = dataX[:,l:t1]
    dataVY = dataY[l:t1]
    dataX = dataX[:,:l]
    dataY = dataY[:l]

    forest = MTForest(dataX,dataY, forest_size=10)
    #forest.Print_Forest()
    #forest.Forest_details()

    dataX, dataY = LoadAbalone()
    dataX = dataX[:,t1:t2]
    dataY = dataY[t1:t2]
   
    forest.Print_Forest()

    for _ in range(2):
        predicted = forest.predict(dataX)
        #rmse = CalcRMSE(predicted*30,dataY*30)
        rmse = CalcRMSE(predicted,dataY)
        print("Root mean squared error: {:.3f}".format(rmse))
        forest.ForestPruning(dataVX,dataVY)

    forest.Forest_details()

    predicted = forest.predict(dataX)
    #rmse = CalcRMSE(predicted*30,dataY*30)
    rmse = CalcRMSE(predicted,dataY)
    print("Root mean squared error: {:.3f}".format(rmse))

    #sorted_predictions, sorted_targets = SortByOut(predicted*30,dataY*30)
    sorted_predictions, sorted_targets = SortByOut(predicted,dataY)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    ax.plot(sorted_predictions, label='Predicted')
    ax.plot(sorted_targets, label='Target')
    ax.legend()
    ax.set(ylabel="Rings", xlabel="Instance #")

    plt.show()

def CalcRMSE(x1, x2):
    rmse = 0
    n = len(x1)
    for i in range(n):
        rmse += (x1[i]-x2[i])**2
    return math.sqrt(rmse/n)

def SortByOut(x, y):
    order = np.argsort(y)
    return x[order], y[order]

def CompareN():
    dataX, dataY = LoadAbalone()
    computational_time = np.zeros(10)
    size = np.zeros(10)
    dataX = dataX[2:6, :]

    for i in tqdm(range(10)):
        computational_time[i] = time.time()
        size[i] = (3*i+1)*20
        data_subset_index = np.random.choice(4000, ((4*i+1)*30))  

        _ = Evolve(29, dataX[:,data_subset_index], dataY[data_subset_index], population_size=30, generations=100, max_poly_order=2)
        computational_time[i] = time.time() - computational_time[i]

    print(computational_time)
    print(size)

def OneDForest():
    l = 1000
    dataX = np.linspace(0,7,l)
    dataY = np.zeros(l)
    '''
    for i in range(l):
        if i < l/3:
            dataY[i] = math.sin(dataX[i])**0.48
        elif i < 2*l/3:
            dataY[i] = math.sqrt(dataX[i])
        else:
            dataY[i] = 1- dataX[i]**4
    '''
    for i in range(l):
        dataY[i] = math.sin(dataX[i]) + np.random.random()*1.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    forest = MTForest(dataX,dataY, forest_size=20)
    #forest.ForestPruning(dataX,dataY)
    forest.Forest_details()
    forest.Print_Forest()
    predicted = forest.predict(dataX)

    ax.plot(dataX,predicted,color = 'red')
    ax.scatter(dataX, dataY)
    plt.show()

def SVRvsPYGASOPE():
    dataY = CreateData6(1,1000)
    dataX = np.linspace(0,2*math.pi, 1000)

    svm = SVR(kernel = 'rbf' , C = 1, gamma = 'auto')
    svm.fit(dataX.reshape(-1, 1), dataY)
    
    predicted_y = svm.predict(dataX.reshape(-1, 1))
    rmse1 = mean_squared_error(predicted_y, dataY)

    fig = plt.figure(figsize=(10,7))

    ax2 = fig.add_subplot(211)
    ax2.scatter(dataX, dataY)
    ax2.plot(dataX, predicted_y, 'r')

    bestIndiv,_ = Evolve(10, dataX, dataY, 
                        population_size=50, generations=200, max_poly_order=5, 
                        Progress=True, debug=False)
    predicted = bestIndiv.out(dataX)
    rmse2 = mean_squared_error(predicted, dataY)

    ax1 = fig.add_subplot(212)
    ax1.scatter(dataX,dataY)
    ax1.plot(dataX,predicted,'r')

    print("\nSVM:\t",rmse1)
    print("PyGasope",rmse2)
    print(bestIndiv)

    plt.show()

###################################Bayesian Optimisation##############################################
def OptimisePYGASOPE(poly):
    class tqdm_skopt(object):
        def __init__(self, **kwargs):
            self._bar = tqdm(**kwargs)
            
        def __call__(self, res):
            self._bar.update()

    space = [Real(0.1,0.3,name='crossover_rate'),
            Real(0.1,0.4, name='mutation_rate'), 
            Real(0.1,0.3, name='elite')]

    @use_named_args(space)
    def objective(**params):
        dataX, dataY = LoadAbalone()
        l = dataY.size
        
        #index = np.random.permutation(l)
        #dataX = dataX[:,index]
        #dataY = dataY[index]
        
        t1 = int(l * 0.6)
        t2 = int(l * 0.8)

        input_train = dataX[:,0:t1]
        target_train = dataY[0:t1]
        input_valid = dataX[:,t1:t2]
        target_valid = dataY[t1:t2]

        bestIndiv,_ = Evolve(10, input_train, target_train,
                             crossover_rate=params['crossover_rate'], 
                             mutation_rate=params['mutation_rate'], 
                             elite=params['elite'],
                             max_poly_order=poly, Progress=False)
        predicted = bestIndiv.out(input_valid)
        mse = mean_squared_error(predicted, target_valid)     
        return mse

    n_calls = 80
    res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=30, 
                         n_jobs=-1, callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")])
    print("\n",res_gp.fun)
    print(res_gp.x)
    #plot_convergence(res_gp)

def OptimiseSVR():
    class tqdm_skopt(object):
        def __init__(self, **kwargs):
            self._bar = tqdm(**kwargs)
            
        def __call__(self, res):
            self._bar.update()

    space = [Integer(1,400,name='C'), 
             Real(1.03,10, prior='log-uniform',name='gamma')]
    #space = [Integer(1,300,name='C')]

    @use_named_args(space)
    def objective(**params):
        dataX, dataY,_ = LoadAbalone()
        dataX = np.transpose(dataX)

        l = dataY.size
        '''
        index = np.random.permutation(l)
        dataX = dataX[index,:]
        dataY = dataY[index]
        '''
        t1 = int(l * 0.6)
        t2 = int(l * 0.8)

        input_train = dataX[0:t1,:]
        target_train = dataY[0:t1]
        input_valid = dataX[t1:t2,:]
        target_valid = dataY[t1:t2]

        svm = SVR(kernel = 'rbf', C = params['C'] , gamma=params['gamma']) # parameters to be tuned
        #svm = SVR(kernel = 'rbf', C = params['C'] , gamma='auto') # parameters to be tuned
        
        svm.fit(input_train, target_train)
        
        predicted = svm.predict(input_valid)
        mse = mean_squared_error(predicted,target_valid) 

        return mse
    n_calls = 80
    res_gp = gp_minimize(objective, space, n_calls=n_calls, n_random_starts=30,
                        callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")])
    print("\n",res_gp.fun)
    print(res_gp.x)
    plot_convergence(res_gp)

###################################Friedman H0 Testing################################################
# see below for critical difference diagrams
# https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
def Friedman():

    tests = 30
    Gasope_mse = np.zeros(tests)
    svm_mse = np.zeros(tests)
    lin_mse = np.zeros(tests)

    for i in tqdm(range(tests)):

        #load, shuffle and split data:
        dataX, dataY = LoadAbalone()
        length = dataY.size
        index = np.random.permutation(length)
        dataX = dataX[:, index]
        dataY = dataY[index]

        l = int(length*0.6)
        t1 = int(length*0.8)
        t2 = length

        trainX = dataX[:,:l]
        trainY = dataY[:l]

        validX = dataX[:,l:t1]
        validY = dataY[l:t1]

        testX = dataX[:,t1:t2]
        testY = dataY[t1:t2]

        ######################Gasope######################################
        bestIndiv,_ = Evolve(10, trainX, trainY, 
                            max_poly_order=3,
                            Progress=False)
        
        predicted = bestIndiv.out(testX)
        Gasope_mse[i] = mean_squared_error(predicted,testY)

        ######################reshape data################################
        trainX = np.transpose(trainX)
        validX = np.transpose(validX)
        testX = np.transpose(testX)

        ######################SVM#########################################
        C = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]
        svm = []
        for j in range(len(C)):
            svm.append(SVR(kernel = 'rbf' , C = C[j], gamma = 'auto'))
            svm[j].fit(trainX, trainY)

        #tune parameter C
        valid_results = []
        for j in range(len(C)):
            predicted = svm[j].predict(validX)
            valid_results.append(mean_squared_error(predicted, validY))
        best_value = valid_results[0]
        best_model = 0
        for j in range(len(C)):
            if valid_results[j] < best_value:
                best_value = valid_results[j]
                best_model = j
        
        predicted = svm[best_model].predict(testX)
        svm_mse[i] =  mean_squared_error(predicted, testY)
        
        ######################LIN REG#########################################
        linreg = linear_model.LinearRegression()
        linreg.fit(trainX, trainY)

        predicted = linreg.predict(testX)
        lin_mse[i] = mean_squared_error(predicted,testY)
        
    ######################Statistical test################################
    #stat, p = wilcoxon(Gasope_mse, svm_mse)
    stat, p = friedmanchisquare(Gasope_mse, svm_mse, lin_mse)
    
    print('\nStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

    print('Gasope avg: ',np.average(Gasope_mse))
    print('SVM avg: ',np.average(svm_mse))
    print('Lin Reg avg: ',np.average(lin_mse))

    # continue with pair wise tests:
    rankSamples(['PyGasope','Gaussian kernel SVM','Linear Regression'],tests,Gasope_mse,svm_mse,lin_mse)

##########################pair wise test & critical difference diagram##########################################
def rankSamples(algo_names, runs, *samples):
    # get number algos
    sets = 0
    for _ in samples:
        sets+=1
    
    average_rank = np.zeros(sets)
    ranks = np.zeros((sets,samples[0].size))

    # get ranks:
    for n in range(samples[0].size):

        compare = np.zeros(sets)
        for i in range(sets):
            compare[i] = samples[i][n]
        
        ranked_compare = rankdata(compare)
        for i in range(sets):
            ranks[i,n] = ranked_compare[i]

    for i in range(sets):
        average_rank[i] = np.average(ranks[i])

    # https://docs.biolab.si/3/data-mining-library/reference/evaluation.cd.html

    cd = Orange.evaluation.compute_CD(average_rank, runs, test='bonferroni-dunn')
    Orange.evaluation.graph_ranks(average_rank, algo_names, cd=cd, width=6, textspace=1.5)
    plt.show()

################################################################################################
# show how generations influences training
def gasope_generations_plot(poly, read=False):
    #fig,ax = plt.subplots(2)
    if not read:
        tests = 200
        repetitions = 30
        increment = 10
        rmse = np.zeros((repetitions,int(tests/increment)))
        scores = []
        j=0
        x = np.zeros(int(tests/increment),dtype=int)

        # load data
        dataX, dataY = LoadAbalone()
        length = dataY.size

        l = int(length*0.8)
        t1 = int(length*0.9)
        t2 = length
        dataX = dataX[:,:l]
        dataY = dataY[:l]

        dataX, dataY = LoadAbalone()
        testX = dataX[:,t1:t2]
        testY = dataY[t1:t2]

        # compute results
        for i in tqdm(range(10,tests+1,increment),desc='outer'):
            tot_score = np.zeros(i)

            for k in tqdm(range(repetitions),desc='inner',leave=False):
                bestIndiv, score = Evolve(10, dataX, dataY, max_poly_order=poly, 
                                        crossover_rate=0.2, mutation_rate=0.35, elite=0.2,
                                        population_size=30, generations=i, Progress=False)
                tot_score += np.sqrt(score)
                predicted = bestIndiv.out(testX)
                rmse[k,j] = CalcRMSE(predicted,testY)
            
            x[j] = int(i)
            scores.append(tot_score/repetitions)
            j+=1

            # save results each iteration incase break
            Data_RMSE = pd.DataFrame(data=rmse, columns=x)
            Data_RMSE.to_csv('generations_vs_rmse.csv')
    else:
        Data_RMSE = pd.read_csv('/home/werner/Desktop/Git/Thesis/PyGasope/generations_vs_rmse.csv')
        Data_RMSE = Data_RMSE.drop('index', axis=1)
    
    '''
    # plot results
    for i in range(x.size):
        label = str(x[i]) + ' generations'
        ax[0].plot(scores[i], label=label)


    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set(ylabel='Training Set RMSE', xlabel='Generation')
    legend = ax[0].legend(loc='upper right')
    '''
    #my_pallete = sns.color_palette('viridis_r',30).as_hex()
    my_pallete = sns.color_palette(['#35b779'])
    #print(my_pallete)
    #print(sns.color_palette('viridis').as_hex())
    #sns.set_palette('viridis')
    
    #ax = sns.pointplot(data = Data_RMSE)
    ax = sns.catplot(data=Data_RMSE, orient='v',kind='box', palette=my_pallete,
                     saturation=0.8, width=0.5, linewidth=0.8,fliersize=2.5)
    #ax = sns.swarmplot(data=Data_RMSE, palette='muted',dodge=True)
    ax.set(ylabel='Validation Set RMSE', xlabel='Evolved for # Generations')

    #fig.tight_layout()
    plt.show()
    #fig.savefig()

# show how popsize influences training
def gasope_popsize_plot(poly, read=False):
    #fig,ax = plt.subplots(2)
    tests = 60
    repetitions = 30
    increment = 5

    if not read:
        rmse = np.zeros((repetitions,int((tests-5)/increment)))
        scores = np.zeros((100, int((tests-5)/increment)))
        j=0
        x = np.zeros(int((tests-5)/increment),dtype=int)

        # load data
        dataX, dataY = LoadAbalone()
        length = dataY.size

        l = int(length*0.8)
        t1 = int(length*0.9)
        t2 = length
        dataX = dataX[:,:l]
        dataY = dataY[:l]

        dataX, dataY = LoadAbalone()
        testX = dataX[:,t1:t2]
        testY = dataY[t1:t2]

        # compute results
        for i in tqdm(range(10,tests+1,increment)):
            tot_score = np.zeros(100)

            for k in tqdm(range(repetitions),leave=False):
                bestIndiv, score = Evolve(10, dataX, dataY, max_poly_order=poly, 
                                        crossover_rate=0.2, mutation_rate=0.35, elite=0.2,
                                        population_size=i, generations=100, Progress=False)
                tot_score += np.sqrt(score)
                predicted = bestIndiv.out(testX)
                rmse[k,j] = CalcRMSE(predicted,testY)
            
            x[j] = int(i)
            scores[:,j] = (tot_score/repetitions)
            j+=1

            # save results each iteration incase break
            Data_RMSE = pd.DataFrame(data=rmse, columns=x)
            Data_RMSE.to_csv('popsize_vs_rmse.csv')

            Data_Training_Scores = pd.DataFrame(data=scores, columns=x)
            Data_Training_Scores.to_csv('training_scores.csv')
    else:
        Data_RMSE = pd.read_csv('/home/werner/Desktop/Git/Thesis/Results/Gasope/popsize_vs_rmse.csv')
        Data_RMSE = Data_RMSE.drop('index', axis=1)
        
        train_scores = pd.read_csv('/home/werner/Desktop/Git/Thesis/Results/Gasope/training_scores.csv')
        #train_scores = train_scores.drop('index', axis=1)


    my_pallete = sns.color_palette('viridis_r',14).as_hex()
    my_pallete = my_pallete[:11]
    #print(my_pallete)

    ax1 = sns.lineplot(data=train_scores,x='run',y='rmse',hue='popsize', palette=my_pallete, legend='full')
    #ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set(ylabel='Training Set RMSE', xlabel='Generation')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[1:], labels=labels[1:],title='Population size:')
    #ax1.legend().set_title('Population size:')
    #legend = ax1.legend(title='Population Size:', loc='upper right')

    #ax = sns.pointplot(data = Data_RMSE)
    ax2 = sns.catplot(data=Data_RMSE, orient='v',kind='box', palette=my_pallete,
                      saturation=0.8, width=0.4, linewidth=0.8
                      )

    ax2.set(ylabel='Validation Set RMSE', xlabel='Evolved with # Individuals')
    # saturation, width, fliersize, linewidth

    #fig.tight_layout()
    #plt.legend(loc='upper center')
    plt.show()
    #fig.savefig()

############################OGASOPE vs PyGASOPE###################################################

# f5 - henon map
def PredictHenon():
    length = 11000
    x, y = HenonMap(length)

    #shuffle data
    index = np.random.permutation(length)
    x = x[:,index]
    y = y[index]

    x_train = x[:,0:10000]
    x_test = x[:,10000:]

    y_train = y[0:10000]
    y_test = y[10000:]

    bestIndiv, _ = Evolve(10, x_train, y_train, max_poly_order=3,
                        population_size=35, generations=100, Progress=True)
    
    predicted = bestIndiv.out(x_test)
    mse = mean_squared_error(predicted,y_test)
    print("\nMean squared error: {:.3f}".format(mse))

    plt.scatter(x_test[0,:], x_test[1,:])
    plt.scatter(predicted, x_test[1,:],marker='+')
    plt.show()

def PredictF1():
    length = 11000
    x, y = function1(length, noise=True)

    #shuffle data
    index = np.random.permutation(length)
    x = x[index]
    y = y[index]

    x_train = x[0:10000]
    x_test = x[10000:]

    y_train = y[0:10000]
    y_test = y[10000:]

    bestIndiv, _ = Evolve(20, x_train, y_train, max_poly_order=5,
                        population_size=30, generations=100, 
                        crossover_rate=0.2 , mutation_rate=0.1, elite=0.1,
                        Progress=True)

    predicted = bestIndiv.out(x_test)
    mse = mean_squared_error(predicted,y_test)
    print("\nMean squared error: {:.6f}".format(mse))

    index = np.argsort(x_test)
    y_test = y_test[index]
    x_test = x_test[index]
    predicted = predicted[index]

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.scatter(x_test,y_test)
    ax.plot(x_test,predicted, c='red')

    plt.show()

################################min leaf samples testing##################################################################
#
def MinLeafs():

    dataX, dataY = LoadDB(4)
    N = len(dataY)
    
    #fig, ax = plt.subplots(2,2,figsize=(10,10))

    l = int(N*0.6)
    t1 = int(N*0.8)
    t2 = N
    validX = dataX[l:t1, :]
    validY = dataY[l:t1]
    testX = dataX[t1:t2, :]
    testY = dataY[t1:t2]
    trainX = dataX[:l, :]
    trainY = dataY[:l]    

    for i in range(5,20,5):
        rmse1 = 0
        rmse2 = 0
        #i = 15

        for j in range(10):
            data_subset_index = np.random.choice(l, i)
            input_data = trainX[data_subset_index,:]
            target_data = trainY[data_subset_index]

            bestIndiv,_ = Evolve(10, input_data, target_data, 
                            population_size=30, generations=100, max_poly_order=3, 
                            Progress=True, debug=False)
        
            predicted_y = bestIndiv.out(input_data)
            rmse1 += mean_squared_error(predicted_y, target_data)

            predicted_y = bestIndiv.out(testX)
            rmse2 += mean_squared_error(predicted_y, testY)

        print("\nGASOPE w N={}:\nRMSE train: {:.4f}\nRMSE test: {:.4f}\n\n".format(i,rmse1/10,rmse2/10)) 

################################preliminary comparison##################################################################
def DBtest():
    #fig , ax= plt.subplots(5,4, figsize=(14,13))
    #ax = ax.flatten()
    
    for i in range(10):
        
        if i!=8:
            continue

        # specify max order
        '''
        o = 3
        if i == 0 or i == 3:
            o=2
        elif i == 2 or i == 5:
            o=1
        '''

        print("For DB", i)
        dataX, dataY = LoadDB(i)
        N = len(dataY)
        print(dataX.shape)

        features = dataX.shape[1]
        if features > 13:
            features = 10

        l = int(N*0.8)
        t1 = int(N*0.8)
        t2 = N
        testX = dataX[t1:t2, :]
        testY = dataY[t1:t2]
        trainX = dataX[:l, :]
        trainY = dataY[:l]
        
        
        svm = SVR(kernel = 'rbf' , C = 10, gamma = 'auto')
        svm.fit(trainX, trainY)
        
        predicted_y = svm.predict(trainX)
        mse1 = mean_squared_error(predicted_y, trainY)
        predicted_y = svm.predict(testX)
        mse2 = mean_squared_error(predicted_y, testY)

        print("SVR:\nTrainset MSE: {:.5f}\nTest set MSE: {:.5f}".format(mse1,mse2))
        #ax[2*i].set_title("SVR DB{}".format(i))
        #ax[2*i].scatter(predicted_y, testY)
        #ax2.plot(dataX, predicted_y, 'r')
        
        terms = 13
        pop_size = 30
        gens = 100

        print("\n{} terms\n{} pop size\n{} generations\n".format(terms,pop_size,gens))
        for o in range(1,4):

            #if o != 1:
            #    continue

            st = time.time()

            bestIndiv,score = Evolve(terms, trainX, trainY, 
                                population_size=pop_size, generations=gens, max_poly_order=o, 
                                Progress=True, debug=False)
            
            predicted_y = bestIndiv.out(trainX)
            mse1 = mean_squared_error(predicted_y, trainY)
            predicted_y = bestIndiv.out(testX)
            mse2 = mean_squared_error(predicted_y, testY)
            print('\n',bestIndiv)

            print("\nGASOPE w o={}:\nTrainset MSE: {:.5f}\nTest set MSE: {:.5f}".format(o,mse1,mse2))
     
            #ax[2*i+1].set_title("GASOPE DB{}".format(i))
            #ax[2*i+1].scatter(predicted_y, testY)
            #ax2.plot(dataX, predicted_y, 'r')

            print('execution time:',time.time()-st)
            print()
            
            plt.plot(score)
            plt.show()   

    #plt.tight_layout()
    #plt.show()
    
def exec_time_test():
    i = 6

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

    # tree depth
    depth = 2
    if i == 1 or i == 5 or i == 7:
        depth = 4
    if i == 6 or i == 8 or i == 9:
        depth = 5

    l = int(N*0.8)
    t1 = int(N*0.8)
    t2 = N
    testX = dataX[t1:t2, :]
    testY = dataY[t1:t2]
    trainX = dataX[:l, :]
    trainY = dataY[:l]

    forest_size = 100
    if i == 1 or i == 6 or i == 7 or i == 8 or i == 9:
        forest_size = 10

    mse = np.zeros((2,forest_size))
    
    params = np.arange(1,forest_size)

    #write_progress(path,1,1,i)

    #params.append(j)

    #st = time.time()

    # train model tree forest
    MTF = MTForest(trainX, trainY, score_in=testX, score_target=testY,
                    tree_depth=depth, forest_size=forest_size, 
                    ga_pop_size=pop_size, ga_gens=gens, ga_terms=terms,
                    max_poly_order=max_poly, min_leaf_samples=min_leafs,
                    Progress=True)

    # test
    mse = MTF.score
    print(mse.T)

    print("ParamValue,Train MSE,Test MSE")
    
    j = 0
    for param in params:
        print("\n{},{},{}".format(param,mse[0,j],mse[1,j]))
        j+=1

    #exec_time.append(time.time()-st)

    #time = time.time() - st

    # filepath,filename, db, run, scores
    #write_file(path,'forestsize', params, i, (rank+1), mse, None)

#PredictF1()

#dataIn,dataTarget = LoadAbalone()
#AnalAbalone(dataIn, dataTarget,print_curve=False)
#CompareN()
#x,y = LoadEnergy(norm=False)
#Analyse(x,y,print_curve=False)

#OneDForest()
#runExamples1D()
#runExamples2D()

#runHousing()
#runAbalone()
#MinLeafs()
#DBtest()
#exec_time_test()

#Friedman()
#OptimiseSVR()
#OptimisePYGASOPE(1)
#OptimisePYGASOPE(2)
#OptimisePYGASOPE(3)

#gasope_generations_plot(2, read=True)
#gasope_popsize_plot(2, read=True)

#gasope_training_plot(3)
#gasope_training_plot2(3)

#PredictHenon()

'''
for i in range(8):
    print("Dataset:",i)
    dataX, dataY = LoadDB(i)
    print("X:", dataX[:3,:])
    print("Y:", dataY[:3])
    print(dataX.shape, dataY.shape,'\n')
'''

