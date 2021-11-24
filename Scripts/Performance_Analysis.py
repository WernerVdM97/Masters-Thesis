import cProfile
import PyGasope
import pstats
import time
from tqdm import tqdm

import Examples
from ArtificialData import *
from pstats import SortKey

#comparing performance of using dictionaries vs np array
def func1(__num_terms, __var_orders, x, *coefficients):
    __num_features = x.shape[0]

    if len(x.shape) == 1: #univariate
        y = 0
        i = 0
        for value in __var_orders.values(): #loop through terms
            y += coefficients[i] * x ** value[0]
            i+=1

    else:   #multivariate            
        y = 0
        i = 0
        for value in __var_orders.values():    #loop through terms
            term = coefficients[i]
            for j in range(x.shape[0]):             #loop through features
                if value[j] != 0:
                    term *= x[j]**value[j]
            y += term
            i += 1

    return y

def func2(__num_terms, __var_orders, x, *coefficients):
    __num_features = x.shape[0]

    if len(x.shape) == 1: #univariate
        y = 0
        i = 0
        for value in __var_orders.values(): #loop through terms
            y += coefficients[i] * x ** value[0]
            i+=1

    else:   #multivariate            
        y = 0
        i = 0
        for value in __var_orders.values():    #loop through terms
            term = coefficients[i]
            for j in range(x.shape[0]):             #loop through features
                if value[j] != 0:
                    term *= x[j]**value[j]
            y += term
            i += 1

    return y

def runComp():
    num_terms = 5
    var_orders1 = {0:[0,0,0,0], 1:[0, 1, 1, 2], 2:[0, 0, 2 , 2], 3:[1, 0, 2, 3], 4:[3, 2, 2, 3] , 5:[1 ,1, 1, 1]}
    #var_orders2 = np.array([[0, 1],[0, 0],[1, 0],[2, 2],[1, 1]])
    data2D,z = TwoDNoise(1000)

    y = 0
    t = time.time()
    for i in tqdm(range(100000)):
        y += func1(num_terms, var_orders1, data2D, 10, 2, -2, 3, -1, -5)
    print("original: ",time.time()-t)
    print(np.sum(y))
    y=0
    t = time.time()
    for i in tqdm(range(100000)):
        y += func2(num_terms, var_orders1, data2D, 10, 2, -2, 3, -1, -5)
    print("new: ",time.time()-t)
    print(np.sum(y))

#cProfile.run('runComp()','npvsdic3')
cProfile.run('Examples.runAbalone()','../Performance/forest')

#TO VIEW, enter in terminal:
#pyprof2calltree -k -i {insert file name}

# %%
