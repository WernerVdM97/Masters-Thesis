#%%
import math
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
import matplotlib 
import matplotlib.pyplot as plt

def F(x,*r):
    y = r[0]*x[0]**3 + r[1]*x[0] + r[2]*x[1]**3 + r[3]*x[1]

    return y

def TwoDNoise(l):
    
    # some artificially noisy data to fit
    r = [-5, 4, -5, 4]

    x = np.linspace(-1, 1,l) + np.random.random(l)/5
    y = np.linspace(0, 0.5,l) + np.random.random(l)/10
    
    data2D = np.array([x,y])

    z = np.zeros(l)
    
    for i in range(l):
        z[i] = F(data2D[:,i], *r) + np.random.random()

    return data2D, z

def CreateData(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(length):
            dataset[i] = math.sin(2*math.pi*i/length)+ np.random.random()/4
    else:
        dataset = np.zeros((num_features,length))
        for i in range(length):
            for j in range(num_features):
                dataset[j,i] = math.sin(2*math.pi*i/length)+ np.random.random()/4

    return dataset

def CreateData2(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(length):
            dataset[i] = math.exp(i*3/length-1.5) + np.random.random()/2

    return dataset

def CreateData3(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(1,length+1):
            dataset[i-1] = 1/(0.2+(2*i/length)) + np.random.random()/2

    return dataset

def CreateData4(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(1,length+1):
            dataset[i-1] = np.log(3*i/length) + np.random.random()/2

    return dataset

def CreateData5(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(length):
            dataset[i] = math.sqrt(3*i/length) + np.random.random()/10

    return dataset

def CreateData6(num_features, length):

    if num_features == 1:
        dataset = np.zeros(length)
        for i in range(length):
            dataset[i] = 1/(1+math.exp((6*i/length)-3)) + np.random.random()/10

    return dataset

def LoadHousing():
    l = 20000
    DataIn = np.zeros((8,l))
    DataTarget = np.zeros(l)
    i=0

    #with open("/home/werner/Desktop/Git/Thesis/PyGasope/db/housing.csv", 'r') as csfile:
    with open("C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/housing.csv", 'r') as csfile:
        csv_reader = csv.DictReader(csfile)
        for row in csv_reader:
            ReadFile = True
            for j in row.values():
                if j == '':
                    ReadFile = False
            
            if ReadFile:
                DataIn[0][i] = row['longitude']
                DataIn[1][i] = row['latitude']
                DataIn[2][i] = row['housing_median_age']
                DataIn[3][i] = row['total_rooms']
                DataIn[4][i] = row['total_bedrooms']
                DataIn[5][i] = row['population']
                DataIn[6][i] = row['households']
                DataIn[7][i] = row['median_income']
                            
                DataTarget[i] = row['median_house_value']
                i+=1
            if i == 20000:
                break
    
    Norm = preprocessing.MinMaxScaler()
    DataIn = Norm.fit_transform(DataIn.T).T
    DataTarget = DataTarget/np.max(DataTarget)
    return DataIn, DataTarget, l

def LoadAbalone():
    l = 4178
    dataset = np.zeros((8, l))
    target = np.zeros(l)
    
    #with open("/home/werner/Desktop/Git/Thesis/PyGasope/db/abalone.csv", 'r') as csvfile:
    with open("C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/abalone.csv", 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        i = 0

        for row in csv_reader:

            dataset[0][i] = row['length']
            dataset[1][i] = row['diameter']
            dataset[2][i] = row['height']
            dataset[3][i] = row['whole weight']
            dataset[4][i] = row['shucked weight']
            dataset[5][i] = row['viscera weight']
            dataset[6][i] = row['shell weight']

            if row['sex'] == 'I' or row['sex'] == 'i':
                dataset[7][i] = 1  

            target[i] = row['target']
            i+=1
    
    Norm = preprocessing.MinMaxScaler()
    dataset = Norm.fit_transform(dataset.T).T
    target = target/np.max(target)

    return dataset, target, l

#X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area
def LoadFF(norm=True):
    l = 505
    dataset = np.zeros((9, l))
    target = np.zeros(l)

    with open("/home/werner/Desktop/Git/Thesis/PyGasope/db/forestfires.csv", 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        i = 0

        for row in csv_reader:
            if float(row['FFMC']) < 60:
                print(i+2)

            dataset[0][i] = row['X']
            dataset[1][i] = row['Y']
            dataset[2][i] = row['FFMC']
            dataset[3][i] = row['DMC']
            dataset[4][i] = row['DC']
            dataset[5][i] = row['ISI']
            dataset[6][i] = row['temp']
            dataset[7][i] = row['RH']
            dataset[8][i] = row['wind']

            target[i] = row['area']
            i+=1

    if norm:    
        Norm = preprocessing.MinMaxScaler()
        dataset = Norm.fit_transform(dataset.T).T
        target = target/np.max(target)

    return dataset, target

#X1,X2,X3,X4,X5,X6,X7,X8,Y1,Y2
def LoadEnergy(norm=True):
    l = 768
    dataset = np.zeros((8, l))
    target = np.zeros(l)

    with open("/home/werner/Desktop/Git/Thesis/PyGasope/db/energy.csv", 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        i = 0

        for row in csv_reader:

            dataset[0][i] = float(row['X1'])
            dataset[1][i] = row['X2']
            dataset[2][i] = row['X3']
            dataset[3][i] = row['X4']
            dataset[4][i] = row['X5']
            dataset[5][i] = row['X6']
            dataset[6][i] = row['X7']
            dataset[7][i] = row['X8']

            target[i] = row['Y2']
            i+=1

    if norm:    
        Norm = preprocessing.MinMaxScaler()
        dataset = Norm.fit_transform(dataset.T).T
        target = target/np.max(target)

    return dataset, target

def Analyse(x, y, Poly = 0, print_curve=False ):
    fig = plt.figure(figsize=(8,12))
    ax = []

    for i in range(x.shape[0]):
        
        axes = np.ones((x.shape[0],20))
        for j in range(x.shape[0]):
            axes[j,:] = np.linspace(0,1,20)
        
        if x.shape[0] != 9:
            k = int(x.shape[0]/2)
            l = 2
        else:
            k = 3
            l = 3

        ax.append(fig.add_subplot(k,l,i+1))
        #ax[i].set_ylim(0,1)
        ax[i].scatter(x[i,:], y, alpha=.6)
        
        if print_curve:
            curve = Poly.out(axes)
            ax[i].plot(axes[i,:], curve, 'r')
    
    plt.tight_layout()
    plt.show()

##############################OGASOPE functions########################################################
def HenonMap(steps, noise=True, scale=True):
    # over bounds [-1,1]
    # x_n+1 = 1 - ax_n**2 + y_n
    # y_n+1 = bx_n

    a =1.4
    b = 0.3

    x = np.zeros(steps)
    y = np.zeros(steps)
    z = np.zeros(steps)

    for i in range(-1,steps-1):

        x[i+1] = 1 - a * x[i]**2 + b*y[i]
        #y[i+1] = b * x[i]
        y[i+1] = x[i+1]
        z[i] = y[i+1]    

        if noise:
            #x[i+1] += np.random.random()*2-1
            #y[i+1] += np.random.random()*2-1
            z[i] += np.random.random()*2-1
    
    #if noise:
        #z += np.random.normal(0,1,steps)

    if scale:
        xmin,xmax = np.min(x),np.max(x)
        x = (x-xmin)/(xmax- xmin)
        x = x*2-1
        y = (y-np.min(y))/(np.max(y)- np.min(y))
        y = y*2-1
        
        z = (z-xmin)/(xmax- xmin)
        z = z*2-1
        
    return np.array([x, y]), z

#f1 - 1D sin
def function1(steps, noise=True, scale=True):
    # over bounds [0,2pi]
    x = np.linspace(0, 2*math.pi, steps)
    y = np.zeros(steps)

    for i in range(steps):
        y[i] = math.sin(x[i])

        if noise:
            y[i] += np.random.random()*2-1

    if scale:
        x = (x- np.min(x))/(np.max(x)-  np.min(x))
        x = x*2-1

    #if noise:
        #y += np.random.normal(0,1,steps)
        
    return np.array([x]),y

# f2 - 2D sin
def function2(steps, noise=True, scale=True):
    # over bounds [0,2pi]
    x = np.linspace(0, 2*math.pi, steps)
    y = np.linspace(0, 2*math.pi, steps)
    z = np.zeros(steps)
    
    for i in range(steps):
        z[i] = math.sin(x[i]) + math.sin(y[i]) 
        if noise:
            z[i] += np.random.random()*2-1


    if scale:
        x = (x- np.min(x))/(np.max(x)-  np.min(x))
        x = x*2-1
        y = (y-np.min(y))/(np.max(y)- np.min(y))
        y = y*2-1

    return np.array([x,y]),z

# f3 - 1D polynomial
def function3(steps, noise=True, scale=True):
    x = np.linspace(-2,2,steps)
    y = np.zeros(steps)
    
    for i in range(steps):
        y[i] = x[i]**5 - 5*x[i]**3 + 4*x[i]
        if noise:
            y[i] += np.random.random()*2-1

    if scale:
        x = (x- np.min(x))/(np.max(x)-  np.min(x))
        x = x*2-1


    return np.array([x]), y

# f4 - 2D polynomial
def function4(steps, noise=True, scale=True):
    x = np.linspace(-2,2,steps)
    y = np.linspace(-2,2,steps)
    z = np.zeros(steps)
    
    for i in range(steps):
        z[i] = x[i]**5 - 5*x[i]**3 + 4*x[i] + \
                y[i]**5 - 5*y[i]**3 + 4*y[i]

        if noise:
            z[i] += np.random.random()*2-1
            
    if scale:
        x = (x- np.min(x))/(np.max(x)-  np.min(x))
        x = x*2-1
        y = (y-np.min(y))/(np.max(y)- np.min(y))
        y = y*2-1

    return np.array([x,y]),z

############################################################################################################33
# final db for testing
def LoadDB(number, linux=True):
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

    if not linux:
        dbs = {0:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/abalone.csv',
            1:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/cali_housing.csv',
            2:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/bos_housing.csv',
            3:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/auto.csv',
            4:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/servo.csv',
            5:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/machine.csv',
            6:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/elevators.csv',
            7:'C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db/CASP.csv',
            8:'',
            9:''}
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