import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
from seaborn.categorical import boxplot

db_names = {0:'abalone',
            1:'cali housing',
            2:'bos housing',
            3:'auto mpg',
            4:'servo',
            5:'machine',
            6:'elevators',
            7:'CASP',
            8:'Friedman Artificial dataset',
            9:'Luis Torgo Artificial dataset'}

##############################
# For linearity
def plot_lin():
    dbs = []
    for j in range(10):
        run = []
        order = []
        trainMSE = []
        testMSE = []
        timings = []

        for i in range(1,31):
           # path = '/home/werner/Desktop/Git/Thesis/CHPC/results/db{}/linearity_{}_db{}.csv'.format(j,i,j)
            path = 'C:/Users/Werner/Documents/GitHub/Thesis/CHPC/results-wo-linbase/db{}/linearity_{}_db{}.csv'.format(j,i,j)

            temp = pd.read_csv(path).values
            exec_times = temp[-1,0:-1]
            temp = temp[:2,:].T
            #print(exec_times)

            #order 1
            run.append(i)
            order.append(1)
            trainMSE.append(temp[1,0])
            testMSE.append(temp[1,1])
            timings.append(float(exec_times[0]))

            #order 2
            run.append(i)
            order.append(2)
            trainMSE.append(temp[2,0])
            testMSE.append(temp[2,1])
            timings.append(float(exec_times[1]))

            #order 3
            run.append(i)
            order.append(3)
            trainMSE.append(temp[3,0])
            testMSE.append(temp[3,1])
            timings.append(float(exec_times[2]))

            #exec time
            #run.append(i)
            #time.append(exec)

        #print(run,order,trainMSE,testMSE,exec_times)
        values = {'Run':run,'MaxPolyOrder':order,'Train MSE':trainMSE, 'Test MSE':testMSE, 'Exec_time':timings}
        #print(values)
        dbs.append(pd.DataFrame(values))

    i=0
    for db in dbs:
        pd.set_option('display.float_format', lambda x: '%.3e' % x)
        print(db_names[i])
        print(db.groupby(['MaxPolyOrder']).mean())
        print(db.groupby(['MaxPolyOrder']).var())
        print()

        my_pallete = sb.color_palette('viridis_r',4).as_hex()
        my_pallete = my_pallete[:3]

        #ax = sb.boxplot(x='MaxPolyOrder', y='Test MSE', data=db, palette=my_pallete)

        ax = sb.catplot(data=db, x='MaxPolyOrder', y='Test MSE', 
                        orient='v',kind='box', palette=my_pallete,
                        saturation=0.8, width=0.5, linewidth=0.8,fliersize=2.5)

        ax.set(title='DB: {}'.format(db_names[i]))
        ax.set(ylabel='Generalisation Set MSE', xlabel='Maximum Polynomial Order')
        plt.tight_layout()
        plt.show()
        i+=1

##############################
# For min leafs
def plot_minleafs():
    dbs = []
    for j in range(10):
        run = []
        order = []
        trainMSE = []
        testMSE = []

        db = pd.DataFrame()

        for i in range(10,31):
            path = '/home/werner/Desktop/Git/Thesis/CHPC/results/db{}/minleafs_{}_db{}.csv'.format(j,i,j)
            temp = pd.read_csv(path)
            #temp = temp[:2,:]
            
            run = np.ones(len(temp),dtype=int)
            temp['Run'] = run*i

            db = pd.concat([db,temp],ignore_index=True)

        dbs.append(db)
        
    i=0
    for db in dbs:
        #print(db.head())
        print('\n',db_names[i])
        print(db.groupby(['ParamValue']).mean())
        
       # ax = sb.boxplot(x='ParamValue', y='Test MSE', data=db)
        #ax.set_title('DB: {}'.format(i))

        my_pallete = sb.color_palette('viridis_r',10).as_hex()
        my_pallete = my_pallete[:8]

        ax = sb.catplot(data=db, x='MaxPolyOrder', y='Test MSE', 
                        orient='v',kind='box', palette=my_pallete,
                        saturation=0.8, width=0.5, linewidth=0.8,fliersize=2.5)

        ax.set(title='DB: {}'.format(db_names[i]))
        ax.set(ylabel='Generalisation Set MSE', xlabel='Maximum Tree Depth')

        plt.tight_layout()
        plt.show()
        
        i+=1

##############################
# For tree depth
def plot_depth():
    dbs = []
    for j in range(10):
        run = []
        order = []
        trainMSE = []
        testMSE = []

        db = pd.DataFrame()

        for i in range(1,31):
            #path = '/home/werner/Desktop/Git/Thesis/CHPC/results/db{}/treedepth_{}_db{}.csv'.format(j,i,j)
            path = 'C:/Users/Werner/Documents/GitHub/Thesis/CHPC/results-wo-linbase/db{}/treedepth_{}_db{}.csv'.format(j,i,j)
            temp = pd.read_csv(path)
            #temp = temp[:2,:]
            
            run = np.ones(len(temp),dtype=int)
            temp['Run'] = run*i

            db = pd.concat([db,temp],ignore_index=True)

        dbs.append(db)
        
    i=0
    for db in dbs:
        #print(db.head())
        pd.set_option('display.float_format', lambda x: '%.3e' % x)
        print('\n',db_names[i])
        print(db.groupby(['ParamValue']).agg([np.mean, np.std]).to_latex())
        
        #conc = db.groupby(['ParamValue']).agg([np.mean, np.std])
        
        #print(conc)
        #print(conc['Train MSE'])

        #conc['Train MSE'].plot(y='mean',yerr='std')
        #conc['Test MSE'].plot(y='mean',yerr='std')

        #ax = sb.boxplot(x='ParamValue', y='Test MSE', data=db)
        #ax.set_title('DB: {}'.format(i))

        my_pallete = sb.color_palette('viridis_r',12).as_hex()
        my_pallete = my_pallete[:10]

        ax = sb.catplot(data=db, x='ParamValue', y=' Exec Time',
                        orient='v', palette=my_pallete, alpha=0.6)
        
        #ax.set(ylabel='Generalisation Set MSE', xlabel='Maximum Tree Depth')
        ax.set(ylabel='Execution Time', xlabel='Maximum Tree Depth')

        plt.tight_layout()
        plt.show()
        i+=1

##############################
# For ensemble size
def plot_size():
    dbs = []
    for j in range(10):
        run = []
        order = []
        trainMSE = []
        testMSE = []

        db = pd.DataFrame()

        for i in range(1,31):
            #path = '/home/werner/Desktop/Git/Thesis/CHPC/results-w-linbase/db{}/forestsize_{}_db{}.csv'.format(j,i,j)
            path = 'C:/Users/Werner/Documents/GitHub/Thesis/CHPC/results-w-linbase/db{}/forestsize_{}_db{}.csv'.format(j,i,j)
            temp = pd.read_csv(path)
            #temp = temp[:2,:]
            
            run = np.ones(len(temp),dtype=int)
            temp['Run'] = run*i

            db = pd.concat([db,temp],ignore_index=True)

        dbs.append(db)
        
    i=0
    for db in dbs:
        #print(db.head())
        pd.set_option('display.float_format', lambda x: '%.3e' % x)
        print('\n',db_names[i])
        print(db.groupby(['ParamValue']).agg([np.mean, np.std]).to_latex())
        
        print('\n',db_names[i])
        print(db.groupby(['ParamValue']).agg([np.mean, np.std]))

        #ax = sb.catplot(x='ParamValue', y='Test MSE', data=db, kind='point')
        #ax.set_title('DB: {}'.format(i))
        #ax = sb.boxplot(x='ParamValue',y='Test MSE', data=db)
        #ax = sb.lineplot(x='ParamValue',y='Test MSE', data=db, hue='Run')

        my_pallete = sb.color_palette('viridis_r',38).as_hex()
        my_pallete = my_pallete[:30]

        #ax = sb.boxplot(x='ParamValue',y='Test MSE', data=db, palette=my_pallete)
        #ax = sb.lineplot(x='ParamValue', y = 'Test MSE' , data = db, hue = 'Run')
        
        ax = sb.catplot(data=db, x='ParamValue', y='Test MSE', 
                        orient='v',kind='box', palette=my_pallete,
                        saturation=0.8, width=0.5, linewidth=0.8,fliersize=2.5)
        
        ax.set(ylabel='Test Set MSE', xlabel='Ensemble Size')

        #plt.plot('ParamValue', 'Train MSE', data=db.groupby(['ParamValue']).mean().reset_index(),color='tab:orange',label='Train')
        #plt.plot('ParamValue', 'Test MSE', data=db.groupby(['ParamValue']).mean().reset_index(), color='tab:blue',label='Test')
        #plt.legend()
       
        plt.title('DB: {}'.format(i))
        plt.show()
        
        i+=1
        

#plot_minleafs()
#plot_lin()
plot_depth()
#plot_size()