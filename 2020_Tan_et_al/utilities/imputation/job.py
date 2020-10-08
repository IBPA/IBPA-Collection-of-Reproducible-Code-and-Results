#!/usr/bin/env python
# job.py
# Currently only working for numerical values


import sys
import pickle
import numpy as np 

import rfimpute


if __name__ == "__main__":

    data_file = sys.argv[1]
    arg_file = sys.argv[2]
    res_file = sys.argv[3]
    
    with open(data_file, "rb") as tmp:
        X = pickle.load(tmp)
    with open(arg_file, "rb") as tmp:
        arg_obj = pickle.load(tmp)

    X_array = np.array(X)
    imp_list = []
    print(arg_obj.vari)
    
    for i in range(len(arg_obj.vari)):
        vari = arg_obj.vari[i]
        misi = arg_obj.misi[i]
        obsi = arg_obj.obsi[i]
        
        if len(misi) == 0:
            imp = np.array([])
            imp_list.append(imp)
            continue
        # TODO
        #      sklearn parameter files

        
        _, p = np.shape(X_array)

        p_train = np.delete(np.arange(p), vari)
        print(p_train)
        X_train = X_array[obsi, :]
        X_train = X_train[:, p_train]
        
        X_test = X_array[misi, :]
        X_test = X_test[:, p_train]
        
        y_train = X_array[obsi, :]
        y_train = y_train[:, vari]

        rf = arg_obj.rf
        imp = rf.fit_predict(X_train, y_train, X_test)
        
        
        X_array[misi,vari] = imp
        imp_list.append(imp)

    arg_obj.results.done = True
    arg_obj.results.imp_list = imp_list
    
    
    with open(res_file, "wb") as tmp:
        pickle.dump(arg_obj.results, tmp)





























