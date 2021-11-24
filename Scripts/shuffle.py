import numpy as np
import pandas as pd

dbs = {
    0:'C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db/abalone.csv',
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

def write_shuffle():
    for i in range(len(dbs)): 

        #print(dbs[i])
        db = pd.read_csv(dbs[i])
        length = len(db)

        name = dbs[i].split('/')[-1][:-4] +"_index"
        print(name)

        #f = open("C:/Users/Werner/Documents/GitHub/Thesis/PyGasope/db_index/" + name +".txt", 'w')
        des = "C:/Users/Werner/Documents/GitHub/Thesis/rsc/dbs/db_index/" + name +".txt"

        indexes = np.zeros((30,length))

        for j in range(30):
            index = np.arange(0,length,1)
            #print(index)
            np.random.shuffle(index)
            #print(index)
            indexes[j,:] = index
            '''
            for c in index:
                f.write(str(c) + ',')
            f.write('\n\n')
            '''

        np.savetxt(des, indexes, fmt='%d')
        #print(indexes)
        #f.close()


    #test = np.loadtxt(des)
    #print(test[0,:])
    #print(test)

def dropNaCali():
    db = pd.read_csv(dbs[1])
    print(len(db))
    print(db.head())

    db = db.dropna()
    print(db.head())
    print(len(db))

    db.to_csv(dbs[1], sep=',', index=False)

#dropNaCali()
write_shuffle()