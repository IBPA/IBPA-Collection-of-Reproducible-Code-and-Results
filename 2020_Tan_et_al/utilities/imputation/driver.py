import numpy as np
import rfimpute

import time

if __name__ == "__main__":
    start_time = time.time()
    data = np.loadtxt('data0.0_50.csv', delimiter = ',')
    #data = data[:,0:4]
    #data = data[0:100,:]
    rfimpute = rfimpute.MissForestImputation()
    result = rfimpute.miss_forest_imputation(data)
    np.savetxt('test.txt',result,delimiter = ',')
    end_time = time.time()
    print(end_time - start_time)
    
