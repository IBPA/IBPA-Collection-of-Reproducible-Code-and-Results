import numpy as np
import missingpy


if __name__ == "__main__":
    data = np.loadtxt('data0.0_50.csv', delimiter = ',')
    #data = data[,0:2]
    imputer = missingpy.MissForest(verbose=1)
    imputer.fit_transform(data)
    
    
    
    