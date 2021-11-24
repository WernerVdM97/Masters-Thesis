import numpy as np
from numpy.lib.function_base import average, delete
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy.stats import rankdata
import Orange

my_pallete = sb.color_palette('viridis_r')
#my_pallete = my_pallete[:11]

# NN results and M5 results are incorrectly indexed. should start at run 0 and end at 29.
def AdjustRuns():
    filename = "C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/M5_results.csv"
    #filename = "C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/NN_results.csv"
    DF = pd.read_csv(filename)

    print(DF)
    DF['Run'] = DF['Run']-1
    print(DF)
    
    DF.to_csv(("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/M5_results.csv"),index=False)
    #DF.to_csv(("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/NN_results.csv"),index=False)


def GroupRuns(model):
    path = "C:/Users/Werner/Documents/GitHub/Thesis/Results/CHPC/Testing/results"
    filename = "/{}_results_run_{}.csv".format(model, 0)
    DF = pd.read_csv((path+filename))

    for i in range(1,30):
        filename = "/{}_results_run_{}.csv".format(model, i)

        #print(filename)
        temp = pd.read_csv((path+filename))

        DF = pd.concat([DF, temp])  

    print(DF)
    DF.to_csv(("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/"+model+"_results.csv"),index=False)


def GroupModels():
    models = ['NN','RF','SVR','M5']

    path = "C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests"
    filename = "/{}_results.csv".format('MTF')
    DF = pd.read_csv((path+filename))

    for model in models:

        filename = "/{}_results.csv".format(model)

        #print(filename)
        temp = pd.read_csv((path+filename))

        DF = pd.concat([DF, temp])  

    
    print(DF)
    DF.to_csv(("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/combined_results.csv"),index=False)


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

    print("Average Rank\n", algo_names,"\n", average_rank)
    # https://docs.biolab.si/3/data-mining-library/reference/evaluation.cd.html

    cd = Orange.evaluation.compute_CD(average_rank, runs, test='bonferroni-dunn')
    Orange.evaluation.graph_ranks(average_rank, algo_names, cd=cd, width=6, textspace=1.5)
    plt.show()


def AnalyseCombinedStats():
    #results = pd.read_csv("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/combined_results.csv")
    results = pd.read_csv("/home/werner/Desktop/Git/Thesis/Results/Final_tests/combined_results.csv")
    #results = results[results['Dataset'] == 'MV']
    
    del results['Run']
    del results['Test_R2']
    del results['Train_R2']
    
    
    #print(results)
    temp = results.groupby("Model").mean()
    #temp = results.groupby("Model").median()
    #del temp['Run']
    print(temp.to_latex())
    
    
    pd.set_option('display.float_format', lambda x: '%.3e' % x)
    
    temp2 = results[['Dataset','Model','Train_MAE', 'Test_MAE']]
    
    temp2 = temp2.groupby(["Dataset","Model"]).agg(['mean','std'])
    #temp2 = results.groupby(["Dataset","Model"]).median()
    #del temp['Run']
    print(temp2.to_latex())

    #g = sb.FacetGrid(results, col='Dataset',hue="Run")
    #g.map(sb.scatterplot, "Model", "Test_RMSE")

    #sb.catplot(data=results, x='Model', y='Test_RMSE', hue='Dataset')
    #plt.show()

    #print(results[results['Model']=='M5'].Test_MAE)

    #group metrics
    Mfive = np.array([])
    Mfive = np.append(Mfive, results[results['Model']=='M5'].Test_RMSE.to_numpy())
    Mfive = np.append(Mfive, results[results['Model']=='M5'].Test_MAE.to_numpy())
    #Mfive = np.append(Mfive, results[results['Model']=='M5'].Test_R2.to_numpy())

    MFensemble = []
    MFensemble = np.append(MFensemble, results[results['Model']=='M5E'].Test_RMSE.to_numpy())
    MFensemble = np.append(MFensemble, results[results['Model']=='M5E'].Test_MAE.to_numpy())
    #MFensemble = np.append(MFensemble, results[results['Model']=='M5E'].Test_R2.to_numpy())

    EnEn = np.array([])
    EnEn = np.append(EnEn, results[results['Model']=='NN'].Test_RMSE.to_numpy())
    EnEn = np.append(EnEn, results[results['Model']=='NN'].Test_MAE.to_numpy()) 
    #EnEn = np.append(EnEn, results[results['Model']=='NN'].Test_R2.to_numpy())

    AreF = np.array([])
    AreF = np.append(AreF, results[results['Model']=='RF'].Test_RMSE.to_numpy())
    AreF = np.append(AreF, results[results['Model']=='RF'].Test_MAE.to_numpy())
    #AreF = np.append(AreF, results[results['Model']=='RF'].Test_R2.to_numpy())
    
    MTef = np.array([])
    MTef = np.append(MTef, results[results['Model']=='MTF'].Test_RMSE.to_numpy())
    MTef = np.append(MTef, results[results['Model']=='MTF'].Test_MAE.to_numpy())
    #MTef = np.append(MTef, results[results['Model']=='MTF'].Test_R2.to_numpy())

    SVare = np.array([])
    SVare = np.append(SVare, results[results['Model']=='SVR'].Test_RMSE.to_numpy())
    SVare = np.append(SVare, results[results['Model']=='SVR'].Test_MAE.to_numpy())
    #SVare = np.append(SVare, results[results['Model']=='SVR'].Test_R2.to_numpy())

    stat, p = friedmanchisquare(Mfive, MFensemble, EnEn, AreF, MTef, SVare)

    '''
    stat, p = friedmanchisquare(results[results['Model']=='M5'].Test_RMSE.to_numpy(), 
                                results[results['Model']=='M5E'].Test_RMSE.to_numpy(),
                                results[results['Model']=='NN'].Test_RMSE.to_numpy(),
                                results[results['Model']=='RF'].Test_RMSE.to_numpy(),
                                results[results['Model']=='MTF'].Test_RMSE.to_numpy(),
                                results[results['Model']=='SVR'].Test_RMSE.to_numpy())
    '''

    print('\nStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

    # continue with pair wise tests:
    #print(results[results['Model']=='M5'].Test_RMSE.to_numpy().shape)
    rankSamples(['M5','M5E','NN', 'RF', 'MTF', 'SVR'],
                600, Mfive, MFensemble, EnEn, AreF, MTef, SVare)
    

    '''
    rankSamples(['M5','M5E','NN', 'RF', 'MTF', 'SVR'],
                300, 
                results[results['Model']=='M5'].Test_RMSE.to_numpy(), 
                results[results['Model']=='M5E'].Test_RMSE.to_numpy(),
                results[results['Model']=='NN'].Test_RMSE.to_numpy(),
                results[results['Model']=='RF'].Test_RMSE.to_numpy(),
                results[results['Model']=='MTF'].Test_RMSE.to_numpy(),
                results[results['Model']=='SVR'].Test_RMSE.to_numpy())
    '''

def AnalyseStatsPerDB():

    #results = pd.read_csv("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/combined_results.csv")
    results = pd.read_csv("/home/werner/Desktop/Git/Thesis/Results/Final_tests/combined_results.csv")

    dbs = ['abalone',
            'cali_housing',
            'bos_housing',
            'auto',
            'servo',
            'machine',
            'elevators',
            'CASP',
            'fried',
            'MV']

    for db in dbs:
        print(db)

        temp = results[results['Dataset'] == db]

        #group metrics
        Mfive = np.array([])
        Mfive = np.append(Mfive, temp[temp['Model']=='M5'].Test_RMSE.to_numpy())
        Mfive = np.append(Mfive, temp[temp['Model']=='M5'].Test_MAE.to_numpy())
        #Mfive = np.append(Mfive, temp[temp['Model']=='M5'].Test_R2.to_numpy())

        MFensemble = []
        MFensemble = np.append(MFensemble, temp[temp['Model']=='M5E'].Test_RMSE.to_numpy())
        MFensemble = np.append(MFensemble, temp[temp['Model']=='M5E'].Test_MAE.to_numpy())
        #MFensemble = np.append(MFensemble, results[results['Model']=='M5E'].Test_R2.to_numpy())

        EnEn = np.array([])
        EnEn = np.append(EnEn, temp[temp['Model']=='NN'].Test_RMSE.to_numpy())
        EnEn = np.append(EnEn, temp[temp['Model']=='NN'].Test_MAE.to_numpy()) 
        #EnEn = np.append(EnEn, results[results['Model']=='NN'].Test_R2.to_numpy())

        AreF = np.array([])
        AreF = np.append(AreF, temp[temp['Model']=='RF'].Test_RMSE.to_numpy())
        AreF = np.append(AreF, temp[temp['Model']=='RF'].Test_MAE.to_numpy())
        #AreF = np.append(AreF, results[results['Model']=='RF'].Test_R2.to_numpy())
        
        MTef = np.array([])
        MTef = np.append(MTef, temp[temp['Model']=='MTF'].Test_RMSE.to_numpy())
        MTef = np.append(MTef, temp[temp['Model']=='MTF'].Test_MAE.to_numpy())
        #MTef = np.append(MTef, results[results['Model']=='MTF'].Test_R2.to_numpy())

        SVare = np.array([])
        SVare = np.append(SVare, temp[temp['Model']=='SVR'].Test_RMSE.to_numpy())
        SVare = np.append(SVare, temp[temp['Model']=='SVR'].Test_MAE.to_numpy())
        #SVare = np.append(SVare, results[results['Model']=='SVR'].Test_R2.to_numpy())

        stat, p = friedmanchisquare(Mfive, MFensemble, EnEn, AreF, MTef, SVare)


        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Same distributions (fail to reject H0)')
        else:
            print('Different distributions (reject H0)')

        # continue with pair wise tests:
        #print(results[results['Model']=='M5'].Test_RMSE.to_numpy().shape)
        rankSamples(['M5','M5E','NN', 'RF', 'MTF', 'SVR'],
                    60, Mfive, MFensemble, EnEn, AreF, MTef, SVare)
        print()


def Analyse():
    #results = pd.read_csv("C:/Users/Werner/Documents/GitHub/Thesis/Results/Final_tests/combined_results.csv")
    results = pd.read_csv("/home/werner/Desktop/Git/Thesis/Results/Final_tests/combined_results.csv")
    #results = results[results['Dataset'] == 'MV']    

    print(results.groupby('Model').mean())

    dbs = ['abalone',
            'cali_housing',
            'bos_housing',
            'auto',
            'servo',
            'machine',
            'elevators',
            'CASP',
            'fried',
            'MV']

    
    for db in dbs:

        temp = results[results['Dataset'] == db]
        
        fig, (ax1, ax2) = plt.subplots(2,1)
        
        sb.boxplot(data=temp, x='Model', y='Test_RMSE', palette=my_pallete, showfliers= False,
                    saturation=0.8, width=0.4, linewidth=0.8, fliersize=2.5, ax=ax1)

        sb.boxplot(data=temp, x='Model', y='Test_MAE', palette=my_pallete, showfliers= False, 
                    saturation=0.8, width=0.4, linewidth=0.8, fliersize=2.5, ax=ax2)
         
        fig.suptitle(db)
        #fig.tight_layout()
    
        #fig.show()
        plt.tight_layout()
        plt.show()


#Analyse()
#AnalyseStatsPerDB()
AnalyseCombinedStats()

#GroupRuns('MTF')
#GroupRuns('SVR')
#GroupRuns('RF')
#AdjustRuns()
#GroupModels()
