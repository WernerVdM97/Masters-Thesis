#%%
import numpy as np
import math
import matplotlib.pyplot as pyplot
from sklearn import linear_model
from sklearn.svm import SVR
from PyGasope import Evolve
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from ArtificialData import *
import os 
from scipy.stats import rankdata
import Orange
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# ASSUMES THE FOLLOWING five MODELS IN ORDER:
# GASOPE w p=1 VS GASOPE w p=2 VS GASOPE w p=3 VS SVR VS LINREG

class Testing:
    def __init__(self, input_data, target_data, 
                 output_filename, no_tests, 
                 resume=False, only_Gasope=False, 
                 notebook=False, poly = 5, metric='mse'
                 ):

        self.__only_Gasope = only_Gasope
        self.__no_samples = target_data.size
        self.__input_data = input_data
        self.__target_data = target_data
        self.__no_tests = no_tests
        self.__output_filename = output_filename
        self.__time = 0
        self.__notebook = notebook
        self.__poly = poly
        self.__metric = metric

        if not only_Gasope:
            self.__no_models = 3
        else:
            self.__no_models = 1

        if not resume:
            self.__init_output()
        
        self.__RunTest(metric=metric)


    # create output file
    def __init_output(self):
        if self.__notebook:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/'+self.__output_filename+'_'+self.__metric+'.txt', 'w')
        else:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/'+self.__output_filename+'_'+self.__metric+'.txt', 'w')
            
        file.write('Test #:\tPyG w p1:\tPyG w p2:\tPyG w p3:\t\tSVR w G:\t\tLin Reg:\t\tTime:\n')
        file.close()


    # write results per test
    def __write(self, run_no, results):
        # assumes results is an array of floats

        if self.__notebook:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/'+self.__output_filename+'_'+self.__metric+'.txt', 'a')
        else:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/'+self.__output_filename+'_'+self.__metric+'.txt', 'a')
            
        file.write(str(run_no))
        file.write('\t\t')

        if not self.__only_Gasope:
            for i in range(self.__no_models+2):
                file.write('{:.7f}\t\t'.format(results[i]))
        else:
            for i in range(self.__no_models):
                file.write('{:.7f}\t\t'.format(results[i]))
        file.write('{:.2f}s\n'.format(time.time()-self.__time))
        file.close()


    # shuffle dataset and split into subsets
    def __ShuffleAndSplitData(self):
        # shuffle data
        index = np.random.permutation(self.__no_samples)
        temp_input = self.__input_data[:,index]
        temp_target = self.__target_data[index]

        if not self.__only_Gasope:
            # split points
            t1 = int(self.__no_samples * 0.8) # start of validation set
            t2 = int(self.__no_samples * 0.9) # start of test set

            # split into training, valid and test set
            self.__input_train_data = temp_input[:,0:t1]
            self.__target_train_data = temp_target[0:t1]

            self.__input_valid_data = temp_input[:,t1:t2]
            self.__target_valid_data = temp_target[t1:t2]

            self.__input_test_data = temp_input[:,t2:]
            self.__target_test_data = temp_target[t2:]

        else: # assumes 12 000 data samples 

            #split points
            t1 = 10000 # start of validation set
            t2 = 11000 # start of validation set
            
            # valid set not needed
            self.__input_train_data = temp_input[:,0:t1]
            self.__target_train_data = temp_target[0:t1]

            self.__input_test_data = temp_input[:,t2:]
            self.__target_test_data = temp_target[t2:]

        del temp_input
        del temp_target


    # needs to be hard coded due to differences in how each model is trained
    # :(
    def __TrainAndTestModels(self, metric):
        
        if not self.__only_Gasope:
            results = np.zeros(self.__no_models+2)

            ######################Gasope######################################
            # max order = 1
            bestIndiv,_ = Evolve(10, self.__input_train_data, 
                                 self.__target_train_data,
                                 crossover_rate=0.3, mutation_rate=0.3, elite=0.15, 
                                 max_poly_order=1, population_size=30, generations=100,
                                 Progress=False)
            predicted = bestIndiv.out(self.__input_test_data)
            if metric == 'mse' or 'rmse':
                results[0] = mean_squared_error(predicted,self.__target_test_data)
            elif metric == 'mae':
                results[0] = mean_absolute_error(predicted,self.__target_test_data)

            # max order = 2
            bestIndiv,_ = Evolve(10, self.__input_train_data, 
                                 self.__target_train_data,
                                 crossover_rate=0.2, mutation_rate=0.35, elite=0.3, 
                                 max_poly_order=2, population_size=30, generations=100,
                                 Progress=False)
            predicted = bestIndiv.out(self.__input_test_data)
            if metric == 'mse'or 'rmse':
                results[1] = mean_squared_error(predicted,self.__target_test_data)
            elif metric == 'mae':
                results[1] = mean_absolute_error(predicted,self.__target_test_data)

            # max order = 3
            bestIndiv,_ = Evolve(10, self.__input_train_data, 
                                 self.__target_train_data,
                                 crossover_rate=0.2, mutation_rate=0.35, elite=0.25, 
                                 max_poly_order=3, population_size=30, generations=100,
                                 Progress=False)
            predicted = bestIndiv.out(self.__input_test_data)
            if metric == 'mse'or 'rmse':
                results[2] = mean_squared_error(predicted,self.__target_test_data)
            elif metric == 'mae':
                results[2] = mean_absolute_error(predicted,self.__target_test_data)

            ######################reshape data################################
            self.__input_train_data = np.transpose(self.__input_train_data)
            self.__input_valid_data = np.transpose(self.__input_valid_data)
            self.__input_test_data = np.transpose(self.__input_test_data)

            ######################SVR#########################################
            # best parameter so far was found to be c = 300, gamma = 1
            # honourable mention C = 200, gamma = 2
            # lets try C = 100 gamma = 0.5
            svm = SVR(kernel = 'rbf' , C = 100, gamma = 0.5) # parameters to be tuned
            svm.fit(self.__input_train_data, self.__target_train_data)
            
            predicted = svm.predict(self.__input_test_data)
            if metric == 'mse'or 'rmse':
                results[3] = mean_squared_error(predicted,self.__target_test_data) 
            elif metric == 'mae':
                results[3] = mean_absolute_error(predicted,self.__target_test_data)

            ######################LIN REG#########################################
            linreg = linear_model.LinearRegression()
            linreg.fit(self.__input_train_data, self.__target_train_data)

            predicted = linreg.predict(self.__input_test_data)
            if metric == 'mse'or 'rmse':
                results[4] = mean_squared_error(predicted,self.__target_test_data)         
            elif metric == 'mae':
                results[4] = mean_absolute_error(predicted,self.__target_test_data)

        else:
            results = np.zeros(self.__no_models)

            # same parameters as OGasope
            bestIndiv,_ = Evolve(20, self.__input_train_data, 
                                 self.__target_train_data,
                                 max_poly_order=self.__poly, # sometimes more or less depending on data
                                 population_size=30, generations=100, 
                                 crossover_rate=0.2 , mutation_rate=0.1, elite=0.1,
                                 Progress=False)

            predicted = bestIndiv.out(self.__input_test_data)
            results[0] = mean_squared_error(predicted,self.__target_test_data)
        
        return results


    # execute MSE test
    def __RunTest(self, metric):
        
        for i in tqdm(range(self.__no_tests)):
            #take time
            self.__time = time.time()

            # shuffle
            self.__ShuffleAndSplitData()

            # train and test
            results = self.__TrainAndTestModels(metric)

            # write
            self.__write(i+1,results)

        print('\n')
        

    # plot critical difference diagram for pairwise tests
    def __rankSamples(self, algo_names, runs, *samples):
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
        #cd = Orange.evaluation.compute_CD(average_rank, runs, test='nemenyi')
        
        print("PyG w p=1:\n{}".format(average_rank[0]))
        print("PyG w p=2:\n{}".format(average_rank[1]))
        print("PyG w p=3:\n{}".format(average_rank[2]))
        print("SVR wG:\n{}".format(average_rank[3]))
        print("LinReg:\n{}".format(average_rank[4]))

        Orange.evaluation.graph_ranks(average_rank, algo_names, cd=cd, width=6, textspace=1.5)
        plt.show()


    # take results and do friedman test and pairwise test
    def ComputeStatistics(self, name):
        if not self.__only_Gasope:
            results, runs = self.ReadResults(name)
            stat, p = friedmanchisquare(results[:,0], results[:,1],results[:,2],
                                        results[:,3],results[:,4])
            print('Statistics=%.3f, p=%.12f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Same distributions (fail to reject H0)')
            else:
                print('Different distributions (reject H0)')
                # continue with pair wise tests:
                self.__rankSamples(['GASOPE with o=1','GASOPE with o=2','GASOPE with o=3','SVR with Gaussian Kernel','Linear Regression'],
                                runs,results[:,0],results[:,1],results[:,2],results[:,3],results[:,4])

            # Single Pairwise Test - Wilcoxon signed
            '''
            stat, p = wilcoxon( results[:,0], results[:,4])
            print('\nStatistics=%.3f, p=%.9f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Same distributions (fail to reject H0)')
            else:
                print('Different distributions (reject H0)')
            '''

        else:
            print("cannot compute statistics on only Gasope")
    
    
    # read in results and print averages and deviation return results
    def ReadResults(self, name):
        results = []
        runs = 0
        templine = ''
        average_time = 0
        if self.__only_Gasope:
            average_mse = 0
            deviation = 0
        else:
            average_mse = np.zeros(5)
            deviation = np.zeros(5)

        if self.__notebook:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/' + name, 'r')
        else:
            file = open('/home/werner/Desktop/Git/Thesis/Results/Gasope/'+ name, 'r')
        
        runs = 0
        for line in file:

            #skip header line
            if runs != 0:
                templine = line.split('\t\t')

                # read results
                if self.__only_Gasope:
                    results.append(float(templine[1]))
                    average_time += float(templine[2][:-3])
                else:
                    temparray = np.zeros(5)
                    temparray[0] = templine[1]
                    temparray[1] = templine[2]
                    temparray[2] = templine[3]
                    temparray[3] = templine[4]
                    temparray[4] = templine[5]
                    average_time += float(templine[6][:-3])

                    results.append(temparray)

            # incremenent run counter
            runs+=1

        # do not add last line as run
        runs-=1
        results = np.array(results)
        average_time = average_time/(runs-1)
        if self.__metric == 'rmse':
            results = np.sqrt(results)

        if self.__only_Gasope:
            average_mse = np.average(results)
            deviation = np.std(results)
        else:
            average_mse[0] = np.average(results[:,0])
            average_mse[1] = np.average(results[:,1])
            average_mse[2] = np.average(results[:,2])
            average_mse[3] = np.average(results[:,3])
            average_mse[4] = np.average(results[:,4])

            deviation[0] = np.std(results[:,0])
            deviation[1] = np.std(results[:,1])
            deviation[2] = np.std(results[:,2])
            deviation[3] = np.std(results[:,3])
            deviation[4] = np.std(results[:,4])
        file.close()

        if self.__only_Gasope:
            print("Gasope:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse,deviation))

        else:
            print("PyG w p=1:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse[0],deviation[0]))
            print("PyG w p=2:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse[1],deviation[1]))
            print("PyG w p=3:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse[2],deviation[2]))
            print("SVR wG:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse[3],deviation[3]))
            print("LinReg:\n{}: {:.6f}\nStd: {:.6f}\n".format(self.__metric,average_mse[4],deviation[4]))

        return results, runs

#%%

'''
x,y = HenonMap(12000, noise=False) # henon
test4 = Testing(x,y,'OnlyGasope_Henon', 0, resume=True, 
                notebook=True, only_Gasope=True, poly=3)
_ = test4.ReadResults()


#%%

print('f1:')
x,y = function1(12000,noise=True) # sin 1D
test1 = Testing(x,y,'OnlyGasope_f1_wNoise', 0, resume=True, 
                notebook=False, only_Gasope=True, poly=3)
_ = test1.ReadResults()

print('f2:')
x,y = function2(12000,noise=True) # sin 2d
test2 = Testing(x,y,'OnlyGasope_f2_wNoise', 0, resume=True, 
                notebook=False, only_Gasope=True, poly=3)
_ = test2.ReadResults()

print('f3:')
x,y = function3(12000,noise=True) # poly 1D
test3 = Testing(x,y,'OnlyGasope_f3_wNoise', 0, resume=True, 
                notebook=False, only_Gasope=True, poly=5)
_ = test3.ReadResults()

print('f4:')
x,y = function4(12000,noise=True) # poly 2D
test4 = Testing(x,y,'OnlyGasope_f4_wNoise', 30, resume=False, 
                notebook=False, only_Gasope=True, poly=5)
_ = test4.ReadResults()
#%%
'''
print('Abalone:')
x,y = LoadAbalone() 
test5 = Testing(x,y,'5Way_Abalone3.0', 0, resume=True, 
                notebook=False, metric='rmse')
test5.ComputeStatistics('5Way_Abalone_all_metrics.txt')

# %%
