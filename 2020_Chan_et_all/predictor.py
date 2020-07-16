# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:47:27 2019

@author: T
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import random
import gc
import subprocess
import talos

from sklearn.preprocessing import MinMaxScaler, StandardScaler, quantile_transform
from sklearn.utils import resample, class_weight
from sklearn.metrics import f1_score, auc, roc_curve, precision_recall_curve, confusion_matrix, balanced_accuracy_score, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras import initializers

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import smogn

from contextlib import redirect_stdout
from collections import Counter

#check GPU usage
# from tensorflow.python.client import device_lib
# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()
# print(device_lib.list_local_devices())

def coeff_determination(y_true, y_pred):
    coef = np.corrcoef(y_pred.flatten(), y_true.flatten())
    rsq = coef[0,1]*coef[0,1]
    return rsq

def f1_calc(model, X_val, y_val):
    lr_probs = model.predict_proba(X_val)
    # keep probabilities for the positive outcome only
    # predict class values
    yhat = model.predict(X_val)
    lr_precision, lr_recall, lr_threshold = precision_recall_curve(y_val, lr_probs)
    lr_f1, lr_auc = f1_score(y_val, yhat.round()), auc(lr_recall, lr_precision)
    return lr_f1, lr_auc, lr_precision, lr_recall, lr_threshold

def consolidate_businesses(predict, actual, label):
    # predict = predict.flatten()
    # actual = actual.flatten()
    # label = np.array(label).flatten()
    
    df = pd.DataFrame(columns=['actual', 'predict', 'label'])
    df['predict'] = predict.flatten()
    df['actual'] = actual.flatten()
    df['label'] = np.array(label).flatten()
    df['resid'] = list(np.array(predict) - np.array(actual))
    
    mmin = []
    mmax = []
    res_mmin = []
    res_mmax = []
    pred = []
    act = []
    lbl = []
    res = []
    for i in np.unique(df['actual']):
        pred.append(np.mean(df.iloc[np.where(df.actual == i)].predict))
        lbl.append(df.iloc[np.where(df.actual == i)].label.iloc[0])
        act.append(np.mean(df.iloc[np.where(df.actual == i)].actual))
        res.append(np.mean(df.iloc[np.where(df.actual == i)].resid))
        res_mmin.append(np.mean(df.iloc[np.where(df.actual == i)].resid) - min(df.iloc[np.where(df.actual == i)].resid))
        res_mmax.append(max(df.iloc[np.where(df.actual == i)].resid) - np.mean(df.iloc[np.where(df.actual == i)].resid))           
        mmin.append(np.mean(df.iloc[np.where(df.actual == i)].predict) - min(df.iloc[np.where(df.actual == i)].predict))
        mmax.append(max(df.iloc[np.where(df.actual == i)].predict) - np.mean(df.iloc[np.where(df.actual == i)].predict))

    df = pd.DataFrame(columns=['actual', 'predict', 'label'])
    df['predict'] = pred
    df['actual'] = act
    df['label'] = lbl
    df['min'] = mmin
    df['max'] = mmax
    df['res_min'] = np.absolute(res_mmin)
    df['res_max'] = res_mmax
    df['resid'] = res
                
    predict = df['predict'].to_numpy()
    actual = df['actual'].to_numpy() 
    label = df['label'].to_numpy()
    min_max = np.array([df['min'], df['max']])
    resid = df['resid'].to_numpy()
    res_min_max = np.array([df['res_min'], df['res_max']])
    
    df.to_csv("df.csv")
    
    return predict, actual, label, min_max, resid, res_min_max


def quantile_normalize(df, input_column_names, output_column_names):
    ID = df.index
    df = df.reset_index()
    df = df.drop("ID", axis=1)      
    df = df.drop(<audit yield>, axis=1)    
    df.to_csv("dftobeqnormed.csv", sep=",", index=False)
    
    subprocess.call(['quantile_normalize.R'], shell=True)
    qnorm_data = pd.read_csv('quantile_normalized.csv')

    qnorm_data["ID"] = ID
    qnorm_data = qnorm_data.set_index("ID")
    
    return qnorm_data

def filtration(aud_data, noaud_data, ci, ret_input_column_names, input_column_names):
    aud_data = aud_data.reset_index()
    concat_list = []
    for feature in ret_input_column_names:
        above = np.where(aud_data[feature] > aud_data[feature].quantile(ci))
        concat_list = np.append(concat_list, above)
        remove = np.unique(concat_list)
        
    removed_pd = aud_data.iloc[remove.astype(int)]
    removed_pd = removed_pd.set_index('ID')

    #drop all businesses related to return entries    
    IDs = aud_data["ID"][remove.astype(int)].unique()
    aud_data = aud_data.set_index('ID')
    
    removed_entries = 0
    for ID in IDs:
        removed_entries = removed_entries + len(aud_data.loc[[ID]])
        aud_data = aud_data.drop(ID) 

    print("Removed {} entries for {} businesses above {}% confidence interval".format(removed_entries, len(IDs), int(ci*100)))

    #remove columns with 0 std 
    remove_columns = aud_data.std()[aud_data.std() == 0].index.values
    print("Removed the following {} features which have 0 std after filtration: ".format(len(remove_columns)))
    print(remove_columns)
    aud_data = aud_data.drop(remove_columns, axis=1)
    noaud_data = noaud_data.drop(remove_columns, axis=1)
    input_column_names = list(set(input_column_names) - set(remove_columns))
    
    print(aud_data.columns)
    
    return aud_data, noaud_data, input_column_names

def one_hot_encode(data, reg_input_column_names):
    for feature in reg_input_column_names:
        data = pd.concat([data, pd.get_dummies(data[feature], prefix=feature)], axis=1)
        data.drop(feature, axis=1, inplace=True)
    return data

def normalization(aud_data, aud_Y, noaud_data, norm_method, predict_method, input_column_names, output_column_names):  
    Y_scaler = []
    
    if norm_method == "quantile":
        aud_pd_X = quantile_normalize(aud_data, input_column_names, output_column_names)
        noaud_pd_X = quantile_normalize(noaud_data, input_column_names, output_column_names)
        
        aud_X = aud_pd_X.to_numpy()
        noaud_X = noaud_pd_X.to_numpy()
        
    elif norm_method == "quantile-sklearn":
        aud_X = quantile_transform(aud_data[input_column_names].to_numpy())
        noaud_X = quantile_transform(noaud_data[input_column_names].to_numpy())
        
    else:
        if norm_method == "minmax":
            X_scaler = MinMaxScaler()
            if predict_method == "regress" or "regress-single":
                Y_scaler = MinMaxScaler()
        
        if norm_method == "zscore":
            X_scaler = StandardScaler()
                            
        aud_X = aud_data[input_column_names].to_numpy()
        noaud_X = noaud_data[input_column_names].to_numpy()
                
        aud_X = X_scaler.fit_transform(aud_X)
        noaud_X = X_scaler.fit_transform(noaud_X)
        
        if predict_method == "regress" or "regress-single":
            aud_Y = Y_scaler.fit_transform(aud_Y)

    return aud_X, noaud_X, aud_Y, Y_scaler

def rank_to_dict(ranks, names, order=1):
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def list_features(key, features, names):
    featurelist = []
    for i in features.sort_values(key, ascending = False).index.values:
        featurelist.append(names[i])
    return featurelist 

def train_test_split_ID(X, y, test_size, ID):
    if (len(ID) != X.shape[0]):
        raise('!')
        
    ID_unique = list(set(ID))
    
    print("{} Unique businesses".format(len(ID_unique)))
    random.shuffle(ID_unique)
    expected_test_num = len(ID)*test_size
    
    idx_train = []
    idx_test = []
    
    for cur_ID in ID_unique:       
        if len(idx_test) < expected_test_num:
            idx_test.extend(np.where(ID == cur_ID)[0].tolist())
        else:
            idx_train.extend(np.where(ID == cur_ID)[0].tolist())
    
    X_train = X.iloc[idx_train].to_numpy()
    X_val = X.iloc[idx_test].to_numpy()
    y_train = y.iloc[idx_train].to_numpy()
    y_val = y.iloc[idx_test].to_numpy()
    ID_train = ID[idx_train]
    ID_test = ID[idx_test]
    
    return X_train, X_val, y_train, y_val, ID_train, ID_test

def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def preprocessing(pd_data, keep_columns, reg_input_column_names):       
    aud_data = pd_data[keep_columns].iloc[np.where(pd_data<audited entries>)] #data with audit info
    noaud_data = pd_data[keep_columns].iloc[np.where(pd_data<unaudited entries>)] #data without audit info    
    aud_data = one_hot_encode(aud_data, reg_input_column_names)    
    noaud_data = one_hot_encode(noaud_data, reg_input_column_names)    
    input_column_names = list(set(aud_data.columns) - set(output_column_names))
    
    noaud_input_column_names = list(set(noaud_data.columns) - set(output_column_names))
    unshared_features = list(set(noaud_input_column_names) - set(input_column_names))
    noaud_data = noaud_data.drop(unshared_features, axis=1)
     
    if predict_method == "regress":
        aud_data = aud_data.iloc[np.where(aud_data[<audit yield>] > 0)] #data with audit info
        
    print("Audit info dataset contains {} entries for {} unique businesses".format(len(aud_data), len(aud_data.groupby("ID").mean())))
    # aud_data.to_csv("prefilter_audit_data.csv")
    aud_data, noaud_data, input_column_names = filtration(aud_data, noaud_data, 0.95, return_input_column_names, input_column_names)
    # aud_data.to_csv("filtered_audit_data.csv")
    
    aud_ID = aud_data.index
    noaud_ID = noaud_data.index
    
    Y = aud_data[output_column_names].to_numpy()
    
    if predict_method == "classify":
        for i in range(len(Y)):
            if Y[i] > <threshold>:
                Y[i] = int(1)
            else:
                Y[i] = int(0)
    
    aud_X, noaud_X, aud_Y, Y_scaler = normalization(aud_data, Y, noaud_data, norm_method, predict_method, input_column_names, output_column_names)   
    
    dataX = pd.DataFrame(data=aud_X, columns=aud_data[input_column_names].columns, index=aud_ID)
    noaud_dataX = pd.DataFrame(data=noaud_X, columns=noaud_data[input_column_names].columns, index=noaud_ID)
    dataY = pd.DataFrame(data=aud_Y, columns=aud_data[output_column_names].columns, index=aud_ID)
    
    return dataX, dataY, noaud_dataX, aud_ID, input_column_names, Y_scaler

def cv_ID(X, y, n_cv, predict_method):
    ID = X.index.values
    ID_unique = list(set(ID))
    random.Random(64).shuffle(ID_unique)
    # print(len(ID_unique))

    if predict_method == "classify":
        grouped_y = dataY.groupby("ID").mean()[<audit yield>]

        pos_IDs = grouped_y.index[grouped_y == 1].to_numpy()
        neg_IDs = grouped_y.index[grouped_y == 0].to_numpy()
        
        np.random.shuffle(pos_IDs)
        np.random.shuffle(neg_IDs)
        pos_cv = np.array_split(pos_IDs, n_cv)
        neg_cv = np.array_split(neg_IDs, n_cv)
    
        cv = []
        for i in range(n_cv):
            cv.append(np.concatenate((pos_cv[i], neg_cv[i])))
        cv = np.array(cv)
        
    else: #regression
        cv = np.array_split(ID_unique, n_cv)
                
    cv_cluster = np.empty((len(ID),1))
    cv_cluster.fill(np.nan)

    for fold in range(n_cv):
        for ID in cv[fold]:
            cv_cluster[np.where(X.index == ID)] = fold

    idx_all = list(range(len(ID)))
                
    idx_test_list = []
    idx_train_list = []
    for cur_cv_idx in range(n_cv):
        cur_idx_test = np.where(cv_cluster == cur_cv_idx)[0].tolist()
        cur_idx_train = list(set(idx_all) - set(cur_idx_test))           
        idx_test_list.append(cur_idx_test)
        idx_train_list.append(cur_idx_train)

    return (idx_train_list, idx_test_list)

def oversample_minority(train_success, train_fail, train):
    if len(train_success) > len(train_fail): #Fail is minority
        fail = np.take(train, train_fail)
        success = np.take(train, train_success)
        success_upsampled = resample(fail,
                                  replace=True, # sample with replacement
                                  n_samples=len(success), # match number in majority class
                                  random_state=42)
        train = np.concatenate([success, success_upsampled])
        return train
        
    else: #Success is minority
        fail = np.take(train, train_fail)
        success = np.take(train, train_success)
        fail_upsampled = resample(success,
                                  replace=True, # sample with replacement
                                  n_samples=len(fail), # match number in majority class
                                  random_state=42)
        train = np.concatenate([fail, fail_upsampled])
        return train
          
def undersample_majority(train_success, train_fail, train):
    # UNDERSAMPLING MAJORITY
    if len(train_success) > len(train_fail):
        fail = np.take(train, train_fail)
        success = np.take(train, train_success)
        success_downsampled = resample(success,
                                  replace=False, # sample with replacement
                                  n_samples=len(fail), # match number in minority class
                                  random_state=42)
        train = np.concatenate([fail, success_downsampled]) 
        return train
    else: #fail is the majority
        fail = np.take(train, train_fail)
        success = np.take(train, train_success)
        fail_downsampled = resample(fail,
                                  replace=False, # sample with replacement
                                  n_samples=len(success), # match number in minority class
                                  random_state=42)
        train = np.concatenate([success, fail_downsampled])    
        return train
    
def sampling(X_train, Y_train, method):
    if method == "classify":
        pre_counter = Counter(Y_train.ravel())
    
        over = SMOTE(sampling_strategy=0.7)
        under = RandomUnderSampler()
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        X_train, Y_train = pipeline.fit_resample(X_train, Y_train)
        
        post_counter = Counter(Y_train)
        # print("Class Distribution: " + str(pre_counter))
        # print("Class Distribution post sampling: " + str(post_counter))
        
        Y_train = Y_train.reshape((len(Y_train),1))
        
    else: #regress
        X_df = pd.DataFrame(data=X_train)
        Y_df = pd.DataFrame(data=Y_train, columns=[<audit yield>])
        data = pd.concat([X_df, Y_df], axis=1)
        
        print(data.isnull().values.any())
        
        try:
            SMOGN_result = smogn.smoter(
                data = data,
                y = <audit yield>  
            )
        except:
            SMOGN_result = data
            
        ## plot y distribution 
        sns.kdeplot(data[<audit yield>], label = "Original")
        sns.kdeplot(SMOGN_result[<audit yield>], label = "Modified")
        plt.show()
        
        print("Before sampling: {} entries".format(len(data)))
        print("After sampling: {} entries".format(len(SMOGN_result)))
        
        # data.to_csv("Prior_sampling.csv")
        # SMOGN_result.to_csv("Post_sampling.csv")
        
        Y_train = SMOGN_result[SMOGN_result.columns[-1]].to_numpy()
        X_train = SMOGN_result.drop(SMOGN_result.columns[-1], axis=1).to_numpy()
        
    return X_train, Y_train

def calculate_class_weights(y_val):
    # Calculate validation _sample_weights_ based on the class distribution of train labels
    val_cls_weights = class_weight.compute_class_weight('balanced', np.unique(y_val), y_val.flatten())
    val_cls_weight_dict = {0: val_cls_weights[0], 1: val_cls_weights[1]}
    val_sample_weights = class_weight.compute_sample_weight(val_cls_weight_dict, y_val.flatten())
    
    return val_cls_weights, val_sample_weights

def sample_weights_acc(X_val, y_val, model, params, val_sample_weights):                
    test_metric_list = model.evaluate(X_val, 
                                      y_val, 
                                      batch_size=params['batch_size'], 
                                      verbose=0,
                                      sample_weight=val_sample_weights)
         
    # metric_list has 3 entries: 
    # [0] val_loss weighted by val_sample_weights, 
    # [1] val_accuracy 
    # [2] val_weighted_accuracy   
    val_acc = test_metric_list[1]
    weighted_val_acc = test_metric_list[2]
    
    return  val_acc, weighted_val_acc

def results(metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, pipeline_mode, predict_mode):
    if predict_mode == "classify":
        # metric_1 = model
        # metric_2 = X_val
        # metric_3 = y_val
        # metric_4 = test_acc
        # metric_5 = weighted_test_acc
        # metric_6 = f1_score
        
        model = metric_1
        X_val = metric_2
        y_val = metric_3
        test_acc = metric_4
        weighted_test_acc = metric_5
        f1 = metric_6
    
        # if pipeline_mode == "validation":
        print("Testing Successes: " + str(len(np.where(y_val == 1)[0])))
        print("Testing Fails: " + str(len(np.where(y_val == 0)[0])))
        print("Testing Accuracy: " + str(test_acc))
        print("Weighted Testing Accuracy: " + str(weighted_test_acc))
        print("Test F1 Score: " + str(f1))
        
        y_val_pred = model.predict(X_val).ravel()
        fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
        auc_keras = auc(fpr, tpr)
    
        if pipeline_mode == "validation":
            # ROC         
            threshold = pd.DataFrame(list(zip(fpr, tpr)), columns=['FPR', 'TPR'])
            threshold['diff'] = threshold['TPR'] - threshold['FPR']            
            optimal_idx = np.argmax(tpr - fpr)
        
            plt.figure(figsize=(5, 5))   
            plt.plot([0, 1], [0, 1], 'k--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color = 'red', marker=(5,1), s=150, zorder=2)
            plt.plot(fpr, tpr, color='b', zorder=1)
            plt.fill_between(fpr, tpr, color='royalblue')
            plt.xlabel('1-Specificity')
            plt.ylabel('Sensitivity')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.title('ROC curve')
            plt.show()
            print('AUROCC=%.3f' % (auc_keras))
            
            # PRECISION RECALL
            baseline_precision = len(np.where(y_val == 1)[0])/(len(np.where(y_val == 0)[0]) + len(np.where(y_val == 1)[0]))
            print('Baseline Precision: {}'.format(baseline_precision))
            lr_f1, lr_auc, lr_precision, lr_recall, lr_threshold = f1_calc(model, X_val, y_val)
            
            threshold = pd.DataFrame(list(zip(lr_recall, lr_precision)), columns=['recall', 'precision'])
            threshold['diff'] = threshold['precision'] - (1-threshold['recall'])
            # print(threshold)
            
            optimal_idx = np.argmax(lr_precision - (1-lr_recall))
            # optimal_threshold = lr_threshold[optimal_idx] 
        
            print('AUPRC=%.3f' % lr_auc)
            plt.figure(figsize=(5, 5))   
            plt.plot(lr_recall, lr_precision, color='b', zorder=1)
            plt.scatter(lr_recall[optimal_idx], lr_precision[optimal_idx], color = 'red', marker=(5,1), s=150, zorder=2)
            plt.fill_between(lr_recall,lr_precision, color='royalblue')
            plt.plot([0,1], [baseline_precision,baseline_precision], color='black', linestyle='--')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.title('Precision-Recall curve')    
            plt.show()
            
        cm = confusion_matrix(y_val, y_val_pred.round())
        
        if pipeline_mode == "validation":
            df_cm = pd.DataFrame(cm, index = ["negative audit", "positive audit"],
                              columns = ["negative audit", "positive audit"])
            plt.figure(figsize = (5,5))
            sns.heatmap(df_cm, annot=True)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
        
        global TP
        global FN
        global FP
        global TN
        global weighted_acc
        
        TN = cm[0][0]
        FN = cm[1][0]
        FP = cm[0][1]
        TP = cm[1][1]
        weighted_acc = weighted_test_acc
        
    else: 
        # metric_1 = selected
        # metric_2 = test_predict_disp
        # metric_3 = test_actual_disp
        # metric_4 = labels
        # metric_5 = min_max
        # metric_6 = resid
        # metric_7 = resid_min_max
        
        selected = metric_1
        test_predict_disp = metric_2
        test_actual_disp = metric_3
        labels = metric_4
        min_max = metric_5
        resid = metric_6
        resid_min_max = metric_7
            
        if pipeline_mode == "validation2":

            ymax = 150000
            xmax = 150000
            xmin = 0
            ymin = -200000
            inc_x = 25000
            inc_y = 100000

            fig1 = plt.figure(figsize=(5, 5))   
            plt.rcParams["axes.grid"] = False
            plt.hlines(0, xmin, xmax, linestyles='dashed')
            print(resid_min_max[:, np.where(labels == 'validation')[0]])
            print(resid[np.where(labels == 'validation')]) 
            g1 = plt.errorbar(test_predict_disp[np.where(labels == 'training')],   resid[np.where(labels == 'training')],   yerr=resid_min_max[:, np.where(labels == 'training')[0]],       ecolor='black',   fmt='.', color='black', linewidth=0.5, capsize=1, capthick=1)
            g2 = plt.errorbar(test_predict_disp[np.where(labels == 'testing')],    resid[np.where(labels == 'testing')],    yerr=resid_min_max[:, np.where(labels == 'testing')[0]],        ecolor='black',   fmt='.', color='black', linewidth=0.5, capsize=1, capthick=1)
            g3 = plt.errorbar(test_predict_disp[np.where(labels == 'validation')], resid[np.where(labels == 'validation')], yerr=resid_min_max[:, np.where(labels == 'validation')[0]],     ecolor='red',     fmt='.', color='red',   linewidth=0.5, capsize=1, capthick=1)  
            plt.legend((g1 ,g3),
                       ('Training/Testing Set', 'Validation Set'),
                        scatterpoints=1,
                        loc='bottom right',
                        ncol=1,
                        fontsize=8)
            ax = fig1.add_subplot()
            import matplotlib
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.xaxis.set_ticks(range(xmin, xmax, inc_x))
            ax.yaxis.set_ticks(range(ymin, ymax, inc_y))
            ax.xaxis.set_tick_params(width=1)
            ax.yaxis.set_tick_params(width=1)
            plt.xticks(np.arange(xmin, xmax, step=inc_x))
            plt.yticks(np.arange(ymin, ymax, step=inc_y))
            plt.title('Residual Plot')
            plt.xlabel('Predicted Audit Yield ($)')
            plt.ylabel('Residuals')
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            plt.show()
            
            ymax = 200000
            xmax = 270000
            xmin = 0
            ymin = 0
            inc_x = 100000            
            inc_y = 50000            
            fig2 = plt.figure(figsize=(5, 5))   
            plt.rcParams["axes.grid"] = False
            g1 = plt.errorbar(test_actual_disp[np.where(labels == 'training')],   test_predict_disp[np.where(labels == 'training')],     yerr=min_max[:, np.where(labels == 'training')[0]],     ecolor='black', fmt='.', color='black',  linewidth=0.5, capsize=1, capthick=1)
            g2 = plt.errorbar(test_actual_disp[np.where(labels == 'testing')],    test_predict_disp[np.where(labels == 'testing')],      yerr=min_max[:, np.where(labels == 'testing')[0]],      ecolor='black', fmt='.', color='black',  linewidth=0.5, capsize=1, capthick=1)
            g3 = plt.errorbar(test_actual_disp[np.where(labels == 'validation')], test_predict_disp[np.where(labels == 'validation')],   yerr=min_max[:, np.where(labels == 'validation')[0]],   ecolor='red',   fmt='.', color='red',    linewidth=0.5, capsize=1, capthick=1)
            plt.legend((g1 ,g3),
                       ('Training/Testing Set', 'Validation Set'),
                        scatterpoints=1,
                        loc='bottom right',
                        ncol=1,
                        fontsize=8)
            ax = fig2.add_subplot()
            plt.plot([0, ymax],[0, ymax], color='r', linestyle='--')
            import matplotlib
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.xaxis.set_ticks(range(xmin, xmax, inc_x))
            ax.yaxis.set_ticks(range(ymin, ymax, inc_y))
            ax.xaxis.set_tick_params(width=1)
            ax.yaxis.set_tick_params(width=1)
            plt.xticks(np.arange(xmin, xmax, step=inc_x))
            plt.yticks(np.arange(ymin, ymax, step=inc_y))
            plt.title('Predicted v. Actual Scatterplot')
            plt.xlabel('Actual Audit Yield ($)')
            plt.ylabel('Predicted Audit Yield ($)')
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            plt.show()
            
def summary_statistics(TP_all, TN_all, FP_all, FN_all, weighted_acc_all):
    from statistics import stdev
    
    PPV_all = [0 for x in range(len(TP_all))]  # precision
    NPV_all = [0 for x in range(len(TP_all))] 
    TPR_all = [0 for x in range(len(TP_all))]  # recall
    TNR_all = [0 for x in range(len(TP_all))]  # specificity
    FDR_all = [0 for x in range(len(TP_all))]  
    F1_all = [0 for x in range(len(TP_all))] 
    
    TP_avg = sum(TP_all)/len(TP_all)
    TP_std = stdev(TP_all)
    TN_avg = sum(TN_all)/len(TN_all)
    TN_std = stdev(TN_all)
    FP_avg = sum(FP_all)/len(FP_all)
    FP_std = stdev(FP_all)
    FN_avg = sum(FN_all)/len(FN_all)
    FN_std = stdev(FN_all)
    
    TP_FP_avg = sum(TP_all+FP_all)/len(TP_all+FP_all)
    TP_FP_std = stdev(TP_all+FP_all)
    TP_FN_avg = sum(TP_all+FN_all)/len(TP_all+FN_all)
    TP_FN_std = stdev(TP_all+FN_all)
    TN_FN_avg = sum(TN_all+FN_all)/len(TN_all+FN_all)
    TN_FN_std = stdev(TN_all+FN_all)
    FP_TN_avg = sum(FP_all+TN_all)/len(FP_all+TN_all)
    FP_TN_std = stdev(FP_all+TN_all)
    
    for i in range(len(TP_all)):
        PPV_all[i] = TP_all[i]/(TP_all[i] + FP_all[i])
        NPV_all[i] = TN_all[i]/(TN_all[i] + FN_all[i])
        TPR_all[i] = TP_all[i]/(TP_all[i] + FN_all[i])
        F1_all[i] = 2*(PPV_all[i]*TPR_all[i])/(PPV_all[i]+TPR_all[i])
        TNR_all[i] = TN_all[i]/(TN_all[i]+FN_all[i])
        FDR_all[i] = FP_all[i]/(FP_all[i] + TP_all[i])
        
        if (PPV_all[i]+TPR_all[i]) == 0:
            F1_all[i] = 0
        if FP_all[i] == 0 and TP_all[i] == 0:
            PPV_all[i] = 0
            FDR_all[i] = 0

    PPV_avg = sum(PPV_all)/len(PPV_all)
    PPV_std = stdev(PPV_all)
    NPV_avg = sum(NPV_all)/len(NPV_all)
    NPV_std = stdev(NPV_all)
    TPR_avg = sum(TPR_all)/len(TPR_all)
    TPR_std = stdev(TPR_all)
    TNR_avg = sum(TNR_all)/len(TNR_all)
    TNR_std = stdev(TNR_all)
    FDR_avg = sum(FDR_all)/len(FDR_all)
    FDR_std = stdev(FDR_all)
    F1_avg = sum(F1_all)/len(F1_all)
    F1_std = stdev(F1_all)
    acc_avg = sum(weighted_acc_all)/len(weighted_acc_all)
    acc_std = stdev(weighted_acc_all)
    
    print("================== SUMMARY ==================")
    print("TN: {:.2f} ± {:.2f}".format(TN_avg, TN_std))
    print("FP: {:.2f} ± {:.2f}".format(FP_avg, FP_std))
    print("FN: {:.2f} ± {:.2f}".format(FN_avg, FN_std))
    print("TP: {:.2f} ± {:.2f}".format(TP_avg, TP_std))
    print("TP+FN: {:.2f} ± {:.2f}".format(TP_FN_avg, TP_FN_std))
    print("TP+FP: {:.2f} ± {:.2f}".format(TP_FP_avg, TP_FP_std))
    print("TN+FN: {:.2f} ± {:.2f}".format(TN_FN_avg, TN_FN_std))
    print("FP+TN: {:.2f} ± {:.2f}".format(FP_TN_avg, FP_TN_std))
    print("Precision: {:.2f} ± {:.2f}".format(PPV_avg, PPV_std))
    print("NPV: {:.2f} ± {:.2f}".format(NPV_avg, NPV_std))
    print("Recall: {:.2f} ± {:.2f}".format(TPR_avg, TPR_std))
    print("Specificity: {:.2f} ± {:.2f}".format(TNR_avg, TNR_std))   
    print("FDR: {:.2f} ± {:.2f}".format(FDR_avg, FDR_std))   
    print("F1: {:.2f} ± {:.2f}".format(F1_avg, F1_std))   
    print("Accuracy: {:.2f} ± {:.2f}".format(acc_avg, acc_std))   
    

def single_regression_model(X_train, y_train, X_test, y_test, params):
    reg_const = 0.00001
    
    current_session = K.get_session();
    #with DeepExplain(session=current_session) as de:  # <-- init DeepExplain context
    model = Sequential()
    model.add(Dense(30, 
                    input_dim=30, 
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(reg_const),
                    activity_regularizer = regularizers.l2(reg_const)))
    model.add(Dropout(0.2))
    model.add(Dense(16, 
                    activation = 'relu',
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(reg_const),
                    activity_regularizer = regularizers.l2(reg_const)))
    model.add(Dropout(0.2))
    model.add(Dense(9, 
                    activation = 'relu',
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(reg_const),
                    activity_regularizer = regularizers.l2(reg_const)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(optimizer = optimizers.Adam(), 
                  loss='mse', 
                  metrics=['mse'])
                  
    earlystop_on = True
    earlystop  = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
    callbacks  = [earlystop] if earlystop_on else []
    out = model.fit(X_train, 
                    y_train, 
                    epochs=1000, 
                    batch_size=16, 
                    callbacks=callbacks,
                    validation_data = (X_test, y_test),
                    verbose=0) 
    
    train_predict = model.predict(X_train)
    train_actual = y_train
    test_predict = model.predict(X_test) 
    test_actual = y_test 
    
    test_predict = test_predict.reshape(len(test_predict),1)
    train_predict = train_predict.reshape(len(train_predict),1)

    train_predict_calc, train_actual_calc, train_label_calc, train_min_max_calc, train_resid_calc, train_res_min_max_calc = consolidate_businesses(train_predict, train_actual, np.full((len(train_predict), 1), 1))
    test_predict_calc, test_actual_disp_calc, test_label_disp_calc, test_min_max_calc, test_resid_calc, test_res_min_max_calc = consolidate_businesses(test_predict, test_actual, np.full((len(test_predict), 1), 1))

    train_predict_disp = Y_scaler.inverse_transform(train_predict)
    train_actual_disp = Y_scaler.inverse_transform(y_train)    
    train_predict_disp, train_actual_disp, train_label_disp, train_min_max, train_resid, train_res_min_max = consolidate_businesses(train_predict_disp, train_actual_disp, np.full((len(train_predict_disp), 1), 1))

    test_predict_disp = Y_scaler.inverse_transform(test_predict)
    test_actual_disp = Y_scaler.inverse_transform(y_test)
    test_predict_disp, test_actual_disp, test_label_disp, test_min_max, test_resid, test_res_min_max = consolidate_businesses(test_predict_disp, test_actual_disp, np.full((len(test_predict_disp), 1), 1))   
    
    train_MSE = np.square(np.subtract(train_actual_calc, train_predict_calc)).mean() 
    train_Rsq = coeff_determination(train_actual_calc, train_predict_calc)
    test_MSE = np.square(np.subtract(test_actual_disp_calc, test_predict_calc)).mean() 
    test_Rsq = coeff_determination(test_actual_disp_calc, test_predict_calc)
    
    cv_train_Rsq.append(train_Rsq)
    cv_test_Rsq.append(test_Rsq)
    
    print("Training R²: {}".format(train_Rsq))    
    print("Testing R²: {}".format(test_Rsq))    
    print("===========================================================================")
    
    results(model, test_predict_disp, test_actual_disp, [], [], [], [], "cv", "regress")
    return out, model, train_predict, train_actual, test_predict, test_actual

    
def regression_model(X_train, y_train, X_test, y_test, params):
    #sample data
    # X_train_sampled, y_train_sampled = sampling(X_train, y_train, 'regress')
    X_train_sampled = X_train
    y_train_sampled = y_train
            
    model = xgb.XGBRegressor(objective = params['objective'], 
                             colsample_bytree = params['colsample_bytree'], 
                             learning_rate = params['learning_rate'],
                             subsample = params['subsample'],
                             max_depth = params['max_depth'], 
                             reg_alpha = params['reg_alpha'], 
                             n_estimators = params['n_estimators'])

    out = model.fit(X_train_sampled, y_train_sampled)
    
    train_predict = model.predict(X_train_sampled)
    train_actual = y_train_sampled
    test_predict = model.predict(X_test) 
    test_actual = y_test 
    #noaud_test_predict = y_scaler.inverse_transform(model.predict(noaud_dataX))
    
    test_predict = test_predict.reshape(len(test_predict),1)
    train_predict = train_predict.reshape(len(train_predict),1)

    train_predict_disp = Y_scaler.inverse_transform(train_predict)
    train_actual_disp = Y_scaler.inverse_transform(y_train)    
    train_predict_disp, train_actual_disp, train_label_disp, train_min_max, train_resid, train_res_min_max = consolidate_businesses(train_predict_disp, train_actual_disp, np.full((len(train_predict_disp), 1), 1))
    test_predict_disp = Y_scaler.inverse_transform(test_predict)
    test_actual_disp = Y_scaler.inverse_transform(y_test)
    test_predict_disp, test_actual_disp, test_label_disp, test_min_max, test_resid, test_res_min_max = consolidate_businesses(test_predict_disp, test_actual_disp, np.full((len(test_predict_disp), 1), 1))   
        
    results(model, test_predict_disp, test_actual_disp, [], [], [], [], "cv", "regress")
        
    return out, model, train_predict, train_actual, test_predict, test_actual


def classify_model(X_train, y_train, X_test, y_test, params):        
    X_train_sampled, y_train_sampled = sampling(X_train, y_train, "classify")

    current_session = K.get_session();
    model = Sequential()
    model.add(Dense(params['first_neuron'], 
                    input_dim=params['input_neurons'], 
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(params['kernel_l2']),
                    activity_regularizer = regularizers.l2(params['activity_l2'])))
    model.add(Dropout(params['dropout']))
    model.add(Dense(19, 
                    activation = params['activation'],
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(params['kernel_l2']),
                    activity_regularizer = regularizers.l2(params['activity_l2'])))
    model.add(Dropout(params['dropout']))
    model.add(Dense(8, 
                    activation = params['activation'],
                    bias_initializer=initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(params['kernel_l2']),
                    activity_regularizer = regularizers.l2(params['activity_l2'])))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[3].get_weights()))
    
    # Compile model
    model.compile(optimizer = optimizers.Adam(),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'],
                  weighted_metrics=['accuracy'])
    
    earlystop_on = True
    earlystop  = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
    callbacks  = [talos.utils.early_stopper(params['epochs'])] if earlystop_on else []
              
    train_cls_weights, train_sample_weights = calculate_class_weights(y_train)
    test_cls_weights, test_sample_weights = calculate_class_weights(y_test)
    
    # baseline probabilities
    # probabilities = [[test_cls_weights[0], test_cls_weights[1]] for _ in range(len(Y_test))]
    # avg_logloss = log_loss(Y_test, probabilities)
    # print('Baseline: Log Loss=%.3f' % (avg_logloss))

    out = model.fit(X_train_sampled, 
                    y_train_sampled, 
                    epochs=params['epochs'], 
                    batch_size=params['batch_size'], 
                    callbacks=callbacks,
                    validation_data = (X_test, y_test, test_sample_weights),
                    verbose=0,
                    class_weight = train_cls_weights)

    train_acc, weighted_train_acc = sample_weights_acc(X_train, y_train, model, params, train_sample_weights)
    test_acc, weighted_test_acc = sample_weights_acc(X_test, y_test, model, params, test_sample_weights)

    train_f1 = f1_score(y_train, model.predict(X_train).round())
    test_f1 = f1_score(y_test, model.predict(X_test).round())

    print("Training Successes: " + str(len(np.where(y_train == 1)[0])))
    print("Training Fails: " + str(len(np.where(y_train == 0)[0])))
    print("Training Accuracy: " + str(train_acc))
    print("Train F1 Score: " + str(train_f1))
        
    plt.plot(out.history['loss'])
    plt.plot(out.history['val_loss'])
    plt.title('Model Loss per Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training set', 'Testing set'], loc='bottom right')
    plt.show()
    
    plt.plot(out.history['accuracy_1'])
    plt.plot(out.history['val_accuracy_1'])
    plt.title('Model Accuracy per Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training set', 'Testing set'], loc='bottom right')
    plt.show()
        
    results(model, X_test, y_test, test_acc, weighted_test_acc, test_f1, [], "cv", "classify")
    
    return out, model

def cross_validation(X, y, kf_idx_train_list, kf_idx_test_list, params, optimizing, predict_mode):
    if predict_mode == "classify":
        test_acc_all = []
        train_acc_all = []
        train_f1_all = []
        test_f1_all = []
        
        TN_all = []
        FP_all = []
        FN_all = []
        TP_all = []
        weighted_acc_all = []
    else:
        cv_train_pred = []
        cv_train_act = []        
        cv_test_pred = []
        cv_test_act = []
        
        global cv_train_Rsq 
        global cv_test_Rsq
        
        cv_train_Rsq = []
        cv_test_Rsq = []
            
    for fold in range(len(kf_idx_train_list)):
        # print("-------------------- Fold " + str(fold+1) + "/"+ str(len(kf_idx_train_list)) +" --------------------")
        train = kf_idx_train_list[fold]
        test = kf_idx_test_list[fold]
        
        X_fold_train = X[train]
        Y_fold_train = y[train]      
        X_fold_test = X[test]
        Y_fold_test = y[test]
                        
        if optimizing == True:
            if predict_mode == "classify":
                scan_object = talos.Scan(x = X_fold_train, 
                                         y = Y_fold_train,
                                         x_val = X_fold_test,
                                         y_val = Y_fold_test,
                                         disable_progress_bar = True,
                                         model = classify_model,
                                         experiment_name = 'classification',
                                         fraction_limit= None, # frac of params to be tested
                                         params = params,
                                         print_params = False)
                report = talos.Reporting(scan_object)
                #report.data.to_csv("Model_params_tested_"+str(len(ranked_feats))+".csv")
                
                best_model = scan_object.best_model(metric='val_accuracy_1', asc=False) #asc False for max problem, True for min problem   
                best_model_index = report.data.loc[report.data['val_accuracy_1'] == np.max(report.data['val_accuracy_1'])].index.values[0]

                TN_all.append(TN)
                TP_all.append(TP)
                FP_all.append(FP)
                FN_all.append(FN)
                weighted_acc_all.append(weighted_acc)
                
                with open('Model_'+ str(len(input_column_names))+'_summary.txt', 'w') as f:
                    with redirect_stdout(f):
                        print(best_model.summary())

            else: #regress
                if predict_mode == "regress-single":
                    out, best_model, y_train_pred, y_train_true, y_test_pred, y_test_true = single_regression_model(X_fold_train, Y_fold_train, X_fold_test, Y_fold_test, params)
                else:
                    out, best_model, y_train_pred, y_train_true, y_test_pred, y_test_true = regression_model(X_fold_train, Y_fold_train, X_fold_test, Y_fold_test, params)
                cv_train_pred.append(y_train_pred)
                cv_train_act.append(y_train_true)
                cv_test_pred.append(y_test_pred)
                cv_test_act.append(y_test_true)
            
        else: #doing feature selection             
            if predict_mode == "classify":
                out, model = classify_model(X_fold_train, Y_fold_train, X_fold_test, Y_fold_test, params)
                #Getting accuracies based on weighted class distribution
                train_cls_weights, train_sample_weights = calculate_class_weights(Y_fold_train)
                test_cls_weights, test_sample_weights = calculate_class_weights(Y_fold_test)
                train_acc, weighted_train_acc = sample_weights_acc(X_fold_train, Y_fold_train, model, params, train_sample_weights)
                test_acc, weighted_test_acc = sample_weights_acc(X_fold_test, Y_fold_test, model, params, test_sample_weights)
                train_f1, train_auc, train_precision, train_recall, lr_threshold = f1_calc(model, X_fold_train, Y_fold_train)
                test_f1, test_auc, test_precision, test_recall, lr_threshold = f1_calc(model, X_fold_test, Y_fold_test)
                
                train_acc_all.append(train_acc)
                test_acc_all.append(weighted_test_acc)
                train_f1_all.append(train_f1)
                test_f1_all.append(test_f1)
                
                with open('Model_'+ str(len(input_column_names))+'_summary.txt', 'a') as f:
                    with redirect_stdout(f):
                        print("-------------------- Fold " + str(fold+1) + "/"+ str(len(kf_idx_train_list)) +" -------------------")
                        print("Training Successes: " + str(len(np.where(Y_fold_train == 1)[0])))
                        print("Training Fails: " + str(len(np.where(Y_fold_train == 0)[0])))
                        print("Training Accuracy: " + str(train_acc))
                        print("Testing Successes: " + str(len(np.where(Y_fold_test == 1)[0])))
                        print("Testing Fails: " + str(len(np.where(Y_fold_test == 0)[0])))
                        print("Testing Accuracy: " + str(weighted_test_acc))
            
            else: #regress
                out, best_model, y_train_pred, y_train_true, y_test_pred, y_test_true = regression_model(X_fold_train, Y_fold_train, X_fold_test, Y_fold_test, params)
                cv_train_pred.append(y_train_pred)
                cv_train_act.append(y_train_true)
                cv_test_pred.append(y_test_pred)
                cv_test_act.append(y_test_true)
                
            K.clear_session()
            gc.collect()
            del model
            
    if optimizing == True:
        if predict_mode == "classify":
            summary_statistics(TP_all, TN_all, FP_all, FN_all, weighted_acc_all)    
            return report, best_model_index, best_model
        elif predict_mode == "regress-single":
            return out, best_model, cv_train_pred, cv_train_act, cv_test_pred, cv_test_act
        else:
            return out, best_model, cv_train_pred, cv_train_act, cv_test_pred, cv_test_act
        
    else:
        if predict_mode == "classify":
            return train_acc_all, test_acc_all, train_f1_all, test_f1_all
        else:
            return out, best_model, cv_train_pred, cv_train_act, cv_test_pred, cv_test_act
        
def evaluate_cv_regression_model(g1_train_pred, g2_train_pred, g3_train_pred, g1_train_act, g2_train_act, g3_train_act, g1_test_pred, g2_test_pred, g3_test_pred, g1_test_act, g2_test_act, g3_test_act, groups):
        cv_train_pred = []
        cv_train_act = []
        cv_test_pred = []
        cv_test_act = []
        cv_train_MSE = []
        cv_test_MSE = []
        cv_train_Rsq = []
        cv_test_Rsq = []
                        
        for i in range(n_cv):
            if groups == 2:
                cv_train_pred.append(np.concatenate((g1_train_pred[i], g2_train_pred[i])))
                cv_train_act.append(np.concatenate((g1_train_act[i], g2_train_act[i])))
                cv_test_pred.append(np.concatenate((g1_test_pred[i], g2_test_pred[i])))
                cv_test_act.append(np.concatenate((g1_test_act[i], g2_test_act[i])))   
            else:
                train_pred = np.concatenate((g1_train_pred[i], g2_train_pred[i], g3_train_pred[i]))
                train_act = np.concatenate((g1_train_act[i], g2_train_act[i], g3_train_act[i]))
                test_pred = np.concatenate((g1_test_pred[i], g2_test_pred[i], g3_test_pred[i]))
                test_act = np.concatenate((g1_test_act[i], g2_test_act[i], g3_test_act[i]))

                global crossval_train_act
                global crossval_test_act
                global crossval_train_pred
                global crossval_test_pred
                global crossval_train_label
                global crossval_test_label
       
                crossval_train_act = train_act
                crossval_train_pred = train_pred
                crossval_test_pred = test_pred
                crossval_test_act = test_act
                crossval_train_label = np.full((len(train_act), 1), 'training') 
                crossval_test_label = np.full((len(test_act), 1), 'testing') 

                cv_train_pred.append(train_pred)
                cv_train_act.append(train_act)
                cv_test_pred.append(test_pred)
                cv_test_act.append(test_act)                   

            cv_train_MSE.append(np.square(np.subtract(cv_train_pred[i], cv_train_act[i])).mean())
            cv_test_MSE.append(np.square(np.subtract(cv_test_pred[i], cv_test_act[i])).mean())
            cv_train_Rsq.append(coeff_determination(cv_train_pred[i], cv_train_act[i]))
            cv_test_Rsq.append(coeff_determination(cv_test_pred[i], cv_test_act[i]))
            
        train_MSE_mean = np.mean(cv_train_MSE)
        train_MSE_std = np.std(cv_train_MSE)
        test_MSE_mean = np.mean(cv_test_MSE)
        test_MSE_std = np.std(cv_test_MSE)
        train_Rsq_mean = np.mean(cv_train_Rsq)
        train_Rsq_std = np.std(cv_train_Rsq)
        test_Rsq_mean = np.mean(cv_test_Rsq)
        test_Rsq_std = np.std(cv_test_Rsq)
                
        print("train R²: {:.2f} ± {:.2f}".format(train_Rsq_mean, train_Rsq_std))
        print("test R²: {:.2f} ± {:.2f}".format(test_Rsq_mean, test_Rsq_std))
        
        return train_MSE_mean, test_MSE_mean, train_Rsq_mean, test_Rsq_mean

def minmax_inversion(test_predict, test_act, label):
    
    test_predict = test_predict.reshape(len(test_predict),1)
    test_act = test_act.reshape(len(test_act),1)
    label = label.reshape(len(label),1)
    consol_test_predict, consol_test_actual, consol_test_label_disp, consol_min_max, consol_resid, consol_res_min_max = consolidate_businesses(test_predict, test_act, label)      
    
    test_predict_disp = Y_scaler.inverse_transform(test_predict)
    test_actual_disp = Y_scaler.inverse_transform(test_act)
    test_predict_disp, test_actual_disp, test_label_disp, test_min_max, test_resid, test_res_min_max = consolidate_businesses(test_predict_disp, test_actual_disp, label)           
    
    return test_predict_disp, test_actual_disp, test_label_disp, test_min_max, test_resid, test_res_min_max, consol_test_predict, consol_test_actual

def evaluate_validation_regression_model(gs_val, pen_val, INDCORP_val, NAICS_val, selected):
    if selected == 0:       
        gsbelow_pred = model1.predict(gs_val.below_X)
        gsbelow_pred = gsbelow_pred.reshape(len(gsbelow_pred),1)            
        gsabove_pred = model2.predict(gs_val.above_X)
        gsabove_pred = gsabove_pred.reshape(len(gsabove_pred),1)            
        gsabove = np.full((len(gsabove_pred), 1), 'above')
        gsbelow = np.full((len(gsbelow_pred), 1), 'below')
        
        gs_pred = np.concatenate((gsbelow_pred, gsabove_pred))
        gs_act = np.concatenate((gs_val.below_y, gs_val.above_y))
        gs_label_1 = np.concatenate((gsbelow, gsabove))
        gs_label_2 = np.full((len(gs_label_1), 1), 'validation') 
        gs_predict_disp, gs_actual_disp, gs_label_disp, gs_min_max, gs_resid, gs_res_min_max, gs_predict, gs_actual = minmax_inversion(gs_pred, gs_act, gs_label_1)
        results(0, gs_predict_disp, gs_actual_disp, gs_label_disp, gs_min_max, gs_resid, gs_res_min_max, "validation", "regress")

        whole_pred = np.concatenate(((crossval_train_pred.flatten(), crossval_test_pred.flatten(), gs_pred.flatten())))
        whole_act = np.concatenate(((crossval_train_act.flatten(), crossval_test_act.flatten(), gs_act.flatten())))
        label = np.concatenate((crossval_train_label.flatten(), crossval_test_label.flatten(), gs_label_2.flatten()))     
        whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, whole_predict, whole_actual = minmax_inversion(whole_pred, whole_act, label)
        results(2, whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, "validation2", "regress")
 
        MAE = np.sum(np.absolute((gs_pred - gs_act)))      
        MSE = np.square(np.subtract(gs_pred, gs_act)).mean()
        Rsq = coeff_determination(gs_pred, gs_act)
 
        MAE_disp = np.sum(np.absolute((gs_predict_disp - gs_actual_disp)))
        MSE_disp = np.square(np.subtract(gs_predict_disp, gs_actual_disp)).mean()
        Rsq_disp = coeff_determination(gs_predict_disp, gs_actual_disp)
      
    if selected == 1:          
        IND_pred = model1.predict(INDCORP_val.group1_X)
        IND_pred = IND_pred.reshape(len(IND_pred),1)            
        CORP_pred = model2.predict(INDCORP_val.group2_X)       
        CORP_pred = CORP_pred.reshape(len(CORP_pred),1)            
        rest_pred = model3.predict(INDCORP_val.group3_X)  
        rest_pred = rest_pred.reshape(len(rest_pred),1)       
        IND = np.full((len(IND_pred), 1), 'g1')
        CORP = np.full((len(CORP_pred), 1), 'g2')
        rest = np.full((len(rest_pred), 1), 'g3')
                
        INDCORP_pred = np.concatenate((IND_pred, CORP_pred, rest_pred))
        INDCORP_act = np.concatenate((INDCORP_val.group1_y, INDCORP_val.group2_y, INDCORP_val.group3_y))
        INDCORP_label_1 = np.concatenate((IND, CORP, rest))        
        INDCORP_label_2 = np.full((len(INDCORP_label_1), 1), 'validation') 
         
        INDCORP_predict_disp, INDCORP_actual_disp, INDCORP_label_disp, INDCORP_min_max, INDCORP_resid, INDCORP_res_min_max, INDCORP_predict, INDCORP_actual = minmax_inversion(INDCORP_pred, INDCORP_act, INDCORP_label_1)
        results(1, INDCORP_predict_disp, INDCORP_actual_disp, INDCORP_label_disp, INDCORP_min_max, INDCORP_resid, INDCORP_res_min_max, "validation", "regress")
        
        whole_pred = np.concatenate(((crossval_train_pred.flatten(), crossval_test_pred.flatten(), INDCORP_pred.flatten())))
        whole_act = np.concatenate(((crossval_train_act.flatten(), crossval_test_act.flatten(), INDCORP_act.flatten())))
        label = np.concatenate((crossval_train_label.flatten(), crossval_test_label.flatten(), INDCORP_label_2.flatten()))     
        whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, whole_predict, whole_actual = minmax_inversion(whole_pred, whole_act, label)
        results(2, whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, "validation2", "regress")
        
        MAE = np.sum(np.absolute((INDCORP_pred - INDCORP_act)))
        MSE = np.square(np.subtract(INDCORP_pred, INDCORP_act)).mean()
        Rsq = coeff_determination(INDCORP_pred, INDCORP_act)
        
        MAE_disp = np.sum(np.absolute((INDCORP_predict_disp - INDCORP_actual_disp)))
        MSE_disp = np.square(np.subtract(INDCORP_predict_disp, INDCORP_actual_disp)).mean()
        Rsq_disp = coeff_determination(INDCORP_predict_disp, INDCORP_actual_disp)
        
    if selected == 2:       
        penhigh_pred = model1.predict(pen_val.above_X)
        penhigh_pred = penhigh_pred.reshape(len(penhigh_pred),1)            
        penlow_pred = model2.predict(pen_val.below_X)
        penlow_pred = penlow_pred.reshape(len(penlow_pred),1)            
        penhigh = np.full((len(penhigh_pred), 1), 'above')
        penlow = np.full((len(penlow_pred), 1), 'below')
        
        pen_pred = np.concatenate((penhigh_pred, penlow_pred))
        pen_act = np.concatenate((pen_val.above_y, pen_val.below_y))
        pen_label_1 = np.concatenate((penhigh, penlow))
        pen_label_2 = np.full((len(pen_label_1), 1), 'validation') 
        pen_predict_disp, pen_actual_disp, pen_label_disp, pen_min_max, pen_resid, pen_res_min_max, pen_predict, pen_actual = minmax_inversion(pen_pred, pen_act, pen_label_1)
        results(2, pen_predict_disp, pen_actual_disp, pen_label_disp, pen_min_max, pen_resid, pen_res_min_max, "validation", "regress")
  
        whole_pred = np.concatenate(((crossval_train_pred.flatten(), crossval_test_pred.flatten(), pen_pred.flatten())))
        whole_act = np.concatenate(((crossval_train_act.flatten(), crossval_test_act.flatten(), pen_act.flatten())))
        label = np.concatenate((crossval_train_label.flatten(), crossval_test_label.flatten(), pen_label_2.flatten()))     
        whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, whole_predict, whole_actual = minmax_inversion(whole_pred, whole_act, label)
        results(2, whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, "validation2", "regress")
          
        MAE = np.sum(np.absolute((pen_pred - pen_act)))
        MSE = np.square(np.subtract(pen_pred, pen_act)).mean()
        Rsq = coeff_determination(pen_pred, pen_act)

        MAE_disp = np.sum(np.absolute((pen_predict_disp - pen_actual_disp)))
        MSE_disp = np.square(np.subtract(pen_predict_disp, pen_actual_disp)).mean()
        Rsq_disp = coeff_determination(pen_predict_disp, pen_actual_disp)
        
    if selected == 3:
        limserv_pred = model1.predict(NAICS_val.group1_X)
        limserv_pred = limserv_pred.reshape(len(limserv_pred),1)            
        mobile_pred = model2.predict(NAICS_val.group2_X)       
        mobile_pred = mobile_pred.reshape(len(mobile_pred),1)            
        other_pred = model3.predict(NAICS_val.group3_X)  
        other_pred = other_pred.reshape(len(other_pred),1)       
        limserv = np.full((len(limserv_pred), 1), 'g1')
        mobile = np.full((len(mobile_pred), 1), 'g2')
        other = np.full((len(other_pred), 1), 'g3')        
        
        NAICS_pred = np.concatenate(((limserv_pred, mobile_pred, other_pred)))
        NAICS_act = np.concatenate((NAICS_val.group1_y, NAICS_val.group2_y, NAICS_val.group3_y))
        NAICS_label_1 = np.concatenate((limserv, mobile, other))        
        NAICS_label_2 = np.full((len(NAICS_label_1), 1), 'validation') 
        NAICS_predict_disp, NAICS_actual_disp, NAICS_label_disp, NAICS_min_max, NAICS_resid, NAICS_res_min_max, NAICS_predict, NAICS_actual = minmax_inversion(NAICS_pred, NAICS_act, NAICS_label_1)
        results(3, NAICS_predict_disp, NAICS_actual_disp, NAICS_label_disp, NAICS_min_max, NAICS_resid, NAICS_res_min_max, "validation", "regress")
        
        whole_pred = np.concatenate(((crossval_train_pred.flatten(), crossval_test_pred.flatten(), NAICS_pred.flatten())))
        whole_act = np.concatenate(((crossval_train_act.flatten(), crossval_test_act.flatten(), NAICS_act.flatten())))
        label = np.concatenate((crossval_train_label.flatten(), crossval_test_label.flatten(), NAICS_label_2.flatten()))
        whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, whole_predict, whole_actual = minmax_inversion(whole_pred, whole_act, label)
        results(3, whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, "validation2", "regress")

        MAE = np.sum(np.absolute((NAICS_predict - NAICS_actual)))
        MSE = np.square(np.subtract(NAICS_predict, NAICS_actual)).mean()
        Rsq = coeff_determination(NAICS_predict, NAICS_actual)
    
        MAE_disp = np.sum(np.absolute((NAICS_predict_disp - NAICS_actual_disp)))
        MSE_disp = np.square(np.subtract(NAICS_predict_disp, NAICS_actual_disp)).mean()
        Rsq_disp = coeff_determination(NAICS_predict_disp, NAICS_actual_disp)
        
    print("Validation MSE: {}".format(MSE))
    print("Validation MAE: {}".format(MAE))            
    print("Validation R²: {}".format(Rsq))   
   
    print("Validation MSE: {}".format(MSE_disp))
    print("Validation MAE: {}".format(MAE_disp))            
    print("Validation R²: {}".format(Rsq_disp))           


def regression_data_partition(gs, pen, INDCORP, NAICS, n_cv, selected, optimization, chosen_params):    
        
    gsbelow_train, gsbelow_test = cv_ID(gs.below_X_df, gs.below_y_df, n_cv, "regress")
    gsabove_train, gsabove_test = cv_ID(gs.above_X_df, gs.above_y_df, n_cv, "regress")
    IND_train, IND_test = cv_ID(INDCORP.group1_X_df, INDCORP.group1_y_df, n_cv, "regress")
    CORP_train, CORP_test = cv_ID(INDCORP.group2_X_df, INDCORP.group2_y_df, n_cv, "regress")
    rest_train, rest_test = cv_ID(INDCORP.group3_X_df, INDCORP.group3_y_df, n_cv, "regress")
    penhigh_train, penhigh_test = cv_ID(pen.above_X_df, pen.above_y_df, n_cv, "regress")
    penlow_train, penlow_test = cv_ID(pen.below_X_df, pen.below_y_df, n_cv, "regress")  
    limserv_train, limserv_test = cv_ID(NAICS.group1_X_df, NAICS.group1_y_df, n_cv, "regress")  
    mobile_train, mobile_test = cv_ID(NAICS.group2_X_df, NAICS.group2_y_df, n_cv, "regress")  
    otherNAICS_train, otherNAICS_test = cv_ID(NAICS.group3_X_df, NAICS.group3_y_df, n_cv, "regress")  
            
    if optimization == 1:
        
        test_params = {"objective":"reg:squarederror",
                        'colsample_bytree': 0.4,
                        'learning_rate': 1,
                        'subsample': 0.3,
                        'max_depth': 2, 
                        'reg_alpha': 5,
                        'n_estimators': 45}
        
        report_gsbelow, model_gsbelow, gsbelow_train_pred, gsbelow_train_act, gsbelow_test_pred, gsbelow_test_act = cross_validation(gs.below_X, gs.below_y, gsbelow_train, gsbelow_test, test_params, True, "regress")        
        report_gsabove, model_gsabove, gsabove_train_pred, gsabove_train_act, gsabove_test_pred, gsabove_test_act = cross_validation(gs.above_X, gs.above_y, gsabove_train, gsabove_test, test_params, True, "regress")

        
        report_IND,  model_IND,  IND_train_pred,  IND_train_act,  IND_test_pred,  IND_test_act  = cross_validation(INDCORP.group1_X, INDCORP.group1_y, IND_train, IND_test, test_params, True, "regress")       
        report_CORP, model_CORP, CORP_train_pred, CORP_train_act, CORP_test_pred, CORP_test_act = cross_validation(INDCORP.group2_X, INDCORP.group2_y, CORP_train, CORP_test, test_params, True, "regress")
        report_rest, model_rest, rest_train_pred, rest_train_act, rest_test_pred, rest_test_act = cross_validation(INDCORP.group3_X, INDCORP.group3_y, rest_train, rest_test, test_params, True, "regress")
          

            
        report_penhigh, model_penhigh, penhigh_train_pred, penhigh_train_act, penhigh_test_pred, penhigh_test_act = cross_validation(pen.above_X, pen.above_y, penhigh_train, penhigh_test, test_params, True, "regress")
        report_penlow,  model_penlow,  penlow_train_pred,  penlow_train_act,  penlow_test_pred,  penlow_test_act  = cross_validation(pen.below_X, pen.below_y, penlow_train, penlow_test, test_params, True, "regress")
    


    
        report_limserv,  model_limserv,  limserv_train_pred,  limserv_train_act,  limserv_test_pred,  limserv_test_act  = cross_validation(NAICS.group1_X, NAICS.group1_y, limserv_train, limserv_test, test_params, True, "regress")
        report_mobile, model_mobile, mobile_train_pred, mobile_train_act, mobile_test_pred, mobile_test_act = cross_validation(NAICS.group2_X, NAICS.group2_y, mobile_train, mobile_test, test_params, True, "regress")
        report_otherNAICS, model_otherNAICS, otherNAICS_train_pred, otherNAICS_train_act, otherNAICS_test_pred, otherNAICS_test_act = cross_validation(NAICS.group3_X, NAICS.group3_y, otherNAICS_train, otherNAICS_test, test_params, True, "regress")

        print("Gross Sales Hypothesis")
        gs_train_MSE, gs_test_MSE, gs_train_Rsq, gs_test_Rsq = evaluate_cv_regression_model(gsbelow_train_pred, gsabove_train_pred, [], 
                                                                                            gsbelow_train_act, gsabove_train_act, [], 
                                                                                            gsbelow_test_pred, gsabove_test_pred, [], 
                                                                                            gsbelow_test_act, gsabove_test_act, [], 
                                                                                            2)
        print("Individual v. Corporations Hypothesis")
        INDCORP_train_MSE, INDCORP_test_MSE, INDCORP_train_Rsq, INDCORP_test_Rsq = evaluate_cv_regression_model(IND_train_pred, CORP_train_pred, rest_train_pred, 
                                                                                                                IND_train_act, CORP_train_act, rest_train_act, 
                                                                                                                IND_test_pred, CORP_test_pred, rest_test_pred, 
                                                                                                                IND_test_act, CORP_test_act, rest_test_act, 
                                                                                                                3)
        print("Penalty Hypothesis")
        pen_train_MSE, pen_test_MSE, pen_train_Rsq, pen_test_Rsq = evaluate_cv_regression_model(penhigh_train_pred, penlow_train_pred, [], 
                                                                                                penhigh_train_act, penlow_train_act, [], 
                                                                                                penhigh_test_pred, penlow_test_pred, [], 
                                                                                                penhigh_test_act, penlow_test_act, [], 
                                                                                                2)
        
        print("NAICS Hypothesis")
        NAICS_train_MSE, NAICS_test_MSE, NAICS_train_Rsq, NAICS_test_Rsq = evaluate_cv_regression_model(limserv_train_pred, mobile_train_pred, otherNAICS_train_pred, 
                                                                                                        limserv_train_act, mobile_train_act, otherNAICS_train_act, 
                                                                                                        limserv_test_pred, mobile_test_pred, otherNAICS_test_pred, 
                                                                                                        limserv_test_act, mobile_test_act, otherNAICS_test_pred, 
                                                                                                        3)
    
        sel_model = [gs_test_Rsq, INDCORP_test_Rsq, pen_test_Rsq, NAICS_test_Rsq]
        selected = sel_model.index(max(sel_model))
        selected = 1
        
        if selected == 0:
            print("==================================================================")
            print("                   GROSS SALES MODEL SELECTED")
            print("==================================================================")
            return 0, test_params, model_gsbelow, model_gsabove, []
        if selected == 1:
            print("==================================================================")
            print("                  INDCORP SALES MODEL SELECTED")  
            print("==================================================================") 
            return 1, test_params, model_IND, model_CORP, model_rest
        if selected == 2:
            print("==================================================================")
            print("                    PENALTY MODEL SELECTED")
            print("==================================================================")
            return 2, test_params, model_penhigh, model_penlow, []
        if selected == 3:
            print("==================================================================")
            print("                     NAICS MODEL SELECTED")    
            print("==================================================================")
            return 3, test_params, model_limserv, model_mobile, model_otherNAICS


    if optimization == 0:
        if selected == 0:
            report_gsbelow, model_gsbelow, gsbelow_train_pred, gsbelow_train_act, gsbelow_test_pred, gsbelow_test_act = cross_validation(gs.below_X, gs.below_y, gsbelow_train, gsbelow_test, chosen_params, True, "regress")
            report_gsabove, model_gsabove, gsabove_train_pred, gsabove_train_act, gsabove_test_pred, gsabove_test_act = cross_validation(gs.above_X, gs.above_y, gsabove_train, gsabove_test, chosen_params, True, "regress")
            gs_train_MSE, gs_test_MSE, gs_train_Rsq, gs_test_Rsq = evaluate_cv_regression_model(gsbelow_train_pred, gsabove_train_pred, [], 
                                                                                                  gsbelow_train_act, gsabove_train_act, [], 
                                                                                                  gsbelow_test_pred, gsabove_test_pred, [], 
                                                                                                  gsbelow_test_act, gsabove_test_act, [], 
                                                                                                  2)
            return gs_train_MSE, gs_test_MSE, gs_train_Rsq, gs_test_Rsq
            
        if selected == 1:
            report_IND,  model_IND,  IND_train_pred,  IND_train_act,  IND_test_pred,  IND_test_act  = cross_validation(INDCORP.group1_X, INDCORP.group1_y, IND_train, IND_test, chosen_params, True, "regress")
            report_CORP, model_CORP, CORP_train_pred, CORP_train_act, CORP_test_pred, CORP_test_act = cross_validation(INDCORP.group2_X, INDCORP.group2_y, CORP_train, CORP_test, chosen_params, True, "regress")
            report_rest, model_rest, rest_train_pred, rest_train_act, rest_test_pred, rest_test_act = cross_validation(INDCORP.group3_X, INDCORP.group3_y, rest_train, rest_test, chosen_params, True, "regress")
         
            INDCORP_train_MSE, INDCORP_test_MSE, INDCORP_train_Rsq, INDCORP_test_Rsq = evaluate_cv_regression_model(IND_train_pred, CORP_train_pred, rest_train_pred, 
                                                                                                                      IND_train_act, CORP_train_act, rest_train_act, 
                                                                                                                      IND_test_pred, CORP_test_pred, rest_test_pred, 
                                                                                                                      IND_test_act, CORP_test_act, rest_test_act, 
                                                                                                                      3)
            return INDCORP_train_MSE, INDCORP_test_MSE, INDCORP_train_Rsq, INDCORP_test_Rsq
            
        if selected == 2:
            report_penhigh, model_penhigh, penhigh_train_pred, penhigh_train_act, penhigh_test_pred, penhigh_test_act = cross_validation(pen.above_X, pen.above_y, penhigh_train, penhigh_test, chosen_params, True, "regress")
            report_penlow,  model_penlow,  penlow_train_pred,  penlow_train_act,  penlow_test_pred,  penlow_test_act  = cross_validation(pen.below_X, pen.below_y, penlow_train, penlow_test, chosen_params, True, "regress")
            pen_train_MSE, pen_test_MSE, pen_train_Rsq, pen_test_Rsq = evaluate_cv_regression_model(penhigh_train_pred, penlow_train_pred, [], 
                                                                                                      penhigh_train_act, penlow_train_act, [], 
                                                                                                      penhigh_test_pred, penlow_test_pred, [], 
                                                                                                      penhigh_test_act, penlow_test_act, [], 
                                                                                                      2)
            return pen_train_MSE, pen_test_MSE, pen_train_Rsq, pen_test_Rsq


        if selected == 3:
            report_limserv,  model_limserv,  limserv_train_pred,  limserv_train_act,  limserv_test_pred,  limserv_test_act  = cross_validation(NAICS.group1_X, NAICS.group1_y, limserv_train, limserv_test, chosen_params, True, "regress")
            report_mobile, model_mobile, mobile_train_pred, mobile_train_act, mobile_test_pred, mobile_test_act = cross_validation(NAICS.group2_X, NAICS.group2_y, mobile_train, mobile_test, chosen_params, True, "regress")
            report_otherNAICS, model_otherNAICS, otherNAICS_train_pred, otherNAICS_train_act, otherNAICS_test_pred, otherNAICS_test_act = cross_validation(NAICS.group3_X, NAICS.group3_y, otherNAICS_train, otherNAICS_test, chosen_params, True, "regress")
            NAICS_train_MSE, NAICS_test_MSE, NAICS_train_Rsq, NAICS_test_Rsq = evaluate_cv_regression_model(limserv_train_pred, mobile_train_pred, otherNAICS_train_pred, 
                                                                                                             limserv_train_act, mobile_train_act, otherNAICS_train_act, 
                                                                                                             limserv_test_pred, mobile_test_pred, otherNAICS_test_pred, 
                                                                                                             limserv_test_act, mobile_test_act, otherNAICS_test_pred, 
                                                                                                             3)
            
            return NAICS_train_MSE, NAICS_test_MSE, NAICS_train_Rsq, NAICS_test_Rsq
    

    

def optimize(param_1, param_2, param_3, param_4, ID, oversample, predict_method, n_cv):
    #classify(X, y, input_column_names, output_column_names, ID, oversample, predict_method, n_cv)
    #regress(gs_tree, pen_tree, INDCORP_tree, NAICS_tree, ID, oversample, predict_method, n_cv)                   

    if predict_method == "classify" or predict_method == "regress-single":
        X = param_1
        y = param_2
        input_column_names = param_3
        output_column_names = param_4
        
        X_df = pd.DataFrame(data=X, columns=input_column_names, index=ID)
        y_df = pd.DataFrame(data=y, columns=output_column_names, index=ID)
        
        test_params = {'activation': ['relu', 'sigmoid'],
                       'first_neuron': [int(X.shape[1]/2)],
                       'hidden_layers': [1, 2],
                       'shapes': ['triangle', 'funnel'],
                       'dropout': [0.2, 0.3, 0.4],
                       'optimizer': ['Adam', 'SGD'],
                       'batch_size': [8, 16, 32, 64],
                       'epochs': [50, 100, 200],
                       'input_neurons': [X.shape[1]]}
        
        if predict_method == "classify":
            test_params = {'hidden_layers': [1],
                           'activation': ['sigmoid'],
                           'first_neuron': [int(X.shape[1])],
                           'shapes': ['funnel'],
                           'optimizer': ['Adam'],
                           'dropout': [0.3],
                           'batch_size': [32],
                           'epochs': [1000],
                           'kernel_l2': [0.0001],
                           'activity_l2': [0.0001],   
                           'lr': [0.0001],
                           'input_neurons': [X.shape[1]]}   
        else:
            test_params = {'hidden_layers': [1],
                           'activation': ['relu'],
                           'first_neuron': [int(X.shape[1])],
                           'shapes': ['funnel'],
                           'optimizer': ['Adam'],
                           'dropout': [0.3],
                           'batch_size': [64],
                           'epochs': [1000],
                           'kernel_l2': [0.0001],
                           'activity_l2': [0.0001],   
                           'lr': [0.0001],
                           'input_neurons': [X.shape[1]]}              
        
        train, test = cv_ID(X_df, y_df, n_cv, predict_method)       
        if predict_method == "classify":
            report, best_model_index, model = cross_validation(X, y, train, test, test_params, optimization, predict_method)
            return report, best_model_index, model
        else:
            out, best_model,  y_train_pred, y_train_true, y_test_pred, y_test_true = cross_validation(X, y, train, test, test_params, optimization, predict_method)
            return best_model, y_train_pred, y_train_true, y_test_pred, y_test_true
    else:
        gs_tree = param_1
        pen_tree = param_2
        INDCORP_tree = param_3
        NAICS_tree = param_4
        
        selected, chosen_params, model1, model2, model3 = regression_data_partition(gs_tree, pen_tree, INDCORP_tree, NAICS_tree, n_cv, "", 1, [])
        return selected, chosen_params, model1, model2, model3
    
predict_method = "regress" #or "regress' or 'regress-single' or 'classify'
norm_method = "minmax" #'zscore' or 'minmax' or 'quantile' or 'quantile-sklearn'
n_cv = 10
optimization = True
validation = True
same_nn_for_all_calcs = True

# load dataset
pd_data = pd.read_csv("Data.csv",sep=",", index_col=0)
pd_data = pd_data[pd_data[<positive audits>]

column_names = pd_data.columns.tolist()
output_column_names = [<names of the target variable>]
remove_column_names = [<names of columns to remove manually>]
return_input_column_names = [<names of return data columns>]
return_input_column_names = list(set(return_input_column_names) - set(remove_column_names))
reg_input_column_names = []
input_column_names = list(set(return_input_column_names).union(set(reg_input_column_names)))
keep_columns = list(set(output_column_names).union(set(input_column_names)))

gs_mean = np.mean(pd_data[pd_data.<audited returns>].<gross sales>)
pen_mean = np.mean(pd_data[pd_data.<audited returns>].<penalty>)
gs_mean_minmax = (gs_mean - np.min(pd_data[pd_data.<audited returns>].<gross sales>))/(np.max(pd_data[pd_data.<audited returns>].<gross sales>) - np.min(pd_data[pd_data.<audited returns>].<gross sales>))
pen_mean_minmax = (pen_mean - np.min(pd_data[pd_data.<audited returns>].<penalty>))/(np.max(pd_data[pd_data.<audited returns>].<penalty>) - np.min(pd_data[pd_data.<audited returns>].<penalty>))

dataX, dataY, noaud_dataX, aud_ID, input_column_names, Y_scaler = preprocessing(pd_data, keep_columns, reg_input_column_names)

ranked_feats = []
ranked_feat_accs = []
ranked_feat_f1 = []
ranked_feat_MSE = []
ranked_feat_Rsq = []

X_train, X_val, y_train, y_val, ID_train, ID_val = train_test_split_ID(dataX, dataY, test_size=0.10, ID=aud_ID)

print(X_train.shape)
print(X_val.shape)

X_train_df = pd.DataFrame(X_train, columns=input_column_names, index=ID_train)
y_train_df = pd.DataFrame(y_train, columns=output_column_names, index=ID_train)
X_val_df = pd.DataFrame(X_val, columns=input_column_names, index=ID_val)
y_val_df = pd.DataFrame(y_val, columns=output_column_names, index=ID_val)

if predict_method == "regress":
    class dual_tree:
        def __init__(self, X_df, y_df, threshold, mean):
            self.below_X = X_df[X_df[threshold] < mean].to_numpy()
            self.below_X_df = X_df[X_df[threshold] < mean]
            self.below_y = y_df[X_df[threshold] < mean].to_numpy()
            self.below_y_df = y_df[X_df[threshold] < mean]
            self.ID_below = X_df[X_df[threshold] < mean].index

            self.above_X = X_df[X_df[threshold] > mean].to_numpy()
            self.above_X_df = X_df[X_df[threshold] > mean]
            self.above_y = y_df[X_df[threshold] > mean].to_numpy()
            self.above_y_df = y_df[X_df[threshold] > mean]
            self.ID_above = X_df[X_df[threshold] > mean].index

    class trio_tree:
        def __init__(self, X_df, y_df, group_1, group_2):    
            self.group1_X = X_df[X_df[group_1] == 1].to_numpy()
            self.group1_X_df = X_df[X_df[group_1] == 1]
            self.group1_y = y_df[X_df[group_1] == 1].to_numpy()
            self.group1_y_df = y_df[X_df[group_1] == 1]
            self.ID_group1 = X_df[X_df[group_1] == 1].index
            
            self.group2_X = X_df[X_df[group_2] == 1].to_numpy()
            self.group2_X_df = X_df[X_df[group_2] == 1]
            self.group2_y = y_df[X_df[group_2] == 1].to_numpy()
            self.group2_y_df = y_df[X_df[group_2] == 1]
            self.ID_group2 = X_df[X_df[group_2] == 1].index
        
            X_df = X_df.reset_index()
            y_df = y_df.reset_index()
            rest_idx = list(set(X_df.index) - set(np.where(X_df.<corporations> == 1)[0]).union(set(np.where(X_df.<individual businesses> == 1)[0])))
            X_df = X_df.iloc[rest_idx]
            y_df = y_df.iloc[rest_idx]
            self.group3_X = X_df.set_index("ID").to_numpy()
            self.group3_X_df = X_df.set_index("ID")
            self.group3_y = y_df.set_index("ID").to_numpy()
            self.group3_y_df = y_df.set_index("ID")
            self.ID_group3 = X_df.set_index("ID").index

    gs_tree = dual_tree(X_train_df, y_train_df, "<gross sales>", gs_mean_minmax)
    pen_tree = dual_tree(X_train_df, y_train_df, "<penalty>", pen_mean_minmax)
    INDCORP_tree = trio_tree(X_train_df, y_train_df, "<individual businesses>", "<corporations>")
    NAICS_tree = trio_tree(X_train_df, y_train_df, "<limited-service restaurants>", "<mobile-service restaurants>")

    gs_tree_test = dual_tree(X_train_df, y_train_df, "<gross sales>", gs_mean_minmax)
    pen_tree_test = dual_tree(X_train_df, y_train_df, "<penalty>", pen_mean_minmax)
    INDCORP_tree_test = trio_tree(X_train_df, y_train_df, "<individual businesses>", "<corporations>")
    NAICS_tree_test = trio_tree(X_train_df, y_train_df, "<limited-service restaurants>", "<mobile-service restaurants>")

    gs_val_tree = dual_tree(X_val_df, y_val_df, "<gross sales>", gs_mean_minmax)
    pen_val_tree = dual_tree(X_val_df, y_val_df, "<penalty>", pen_mean_minmax)
    INDCORP_val_tree = trio_tree(X_val_df, y_val_df, "<individual businesses>", "<corporations>")
    NAICS_val_tree = trio_tree(X_val_df, y_val_df, "<limited-service restaurants>", "<mobile-service restaurants>")

while len(input_column_names) > 0: #remove one feature at a time
        
    if optimization == True:
        print("OPTIMIZING...")
        if predict_method == "classify":
            report, best_model_index, model = optimize(X_train, y_train, input_column_names, output_column_names, ID_train, True, predict_method, n_cv)
        elif predict_method == "regress-single":
            model, y_train_pred, y_train_true, y_test_pred, y_test_true = optimize(X_train, y_train, input_column_names, output_column_names, ID_train, True, predict_method, n_cv)            
        else: #regression
            selected, chosen_params, model1, model2, model3 = optimize(gs_tree, pen_tree, INDCORP_tree, NAICS_tree, input_column_names, True, predict_method, n_cv)

        if same_nn_for_all_calcs == True:
            optimization = False   
            
        input("         Press ENTER to validate on validation set...")
    
    if validation == True:
        print("TESTING ON VALIDATION SET...")
        if predict_method == "classify":
            print("Validation Successes: " + str(len(np.where(y_val == 1)[0])))
            print("Validation Fails: " + str(len(np.where(y_val == 0)[0])))
            print("Validation Baseline Ratio: " + str(len(np.where(y_val == 1)[0])/(len(np.where(y_val == 0)[0]) + len(np.where(y_val == 1)[0]))))

            y_val_pred = model.predict(X_val).ravel()
            acc = accuracy_score(y_val, y_val_pred.round())
            weighted_acc = balanced_accuracy_score(y_val, y_val_pred.round())
            f1 = f1_score(y_val, y_val_pred.round())
            print(model.summary())
            results(model, X_val, y_val, acc, weighted_acc, f1, [], "validation", "classify")
            
        elif predict_method == "regress-single": 
            print(model.summary())

            print("Training R²: {:.2f} ± {:.2f}".format(np.mean(cv_train_Rsq),np.std(cv_train_Rsq)))
            print("Testing R²: {:.2f} ± {:.2f}".format(np.mean(cv_test_Rsq),np.std(cv_test_Rsq)))

            y_train_pred = np.vstack(y_train_pred)
            y_test_pred = np.vstack(y_test_pred)
            y_train_true = np.vstack(y_train_true)
            y_test_true = np.vstack(y_test_true)

            train_label = np.full((len(y_train_true), 1), 'training') 
            test_label = np.full((len(y_test_true), 1), 'testing')
            val_pred = model.predict(X_val)
            val_label = np.full((len(y_val), 1), 'validation') 

            val_MSE = np.square(np.subtract(y_val, val_pred)).mean() 
            val_Rsq = coeff_determination(y_val, val_pred)
            
            label = np.concatenate((train_label.flatten(), test_label.flatten(), val_label.flatten()))
            val_predict_disp, val_actual_disp, val_label, val_min_max, val_resid, val_res_min_max, val_predict, val_actual = minmax_inversion(val_pred, y_val, val_label)
            
            val_MSE = np.square(np.subtract(y_val, val_pred)).mean() 
            val_MAE = np.sum(np.absolute((y_val, val_pred)))
            val_Rsq = coeff_determination(y_val, val_pred)
            val_disp_MSE = np.square(np.subtract(val_actual_disp, val_predict_disp)).mean() 
            val_disp_MAE = np.sum(np.absolute((val_actual_disp, val_predict_disp)))
            val_dsp_Rsq = coeff_determination(val_actual_disp, val_predict_disp)
 
            print("Validation MSE: {}".format(val_MSE))
            print("Validation MAE: {}".format(val_MAE))            
            print("Validation R²: {}".format(val_Rsq))   
           
            print("Validation MSE: {}".format(val_disp_MSE))
            print("Validation MAE: {}".format(val_disp_MAE))            
            print("Validation R²: {}".format(val_dsp_Rsq))   
                
            whole_pred = np.concatenate(((y_train_pred, y_test_pred, val_pred)))
            whole_act = np.concatenate(((y_train_true, y_test_true, y_val)))    
            whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, whole_predict, whole_actual = minmax_inversion(whole_pred, whole_act, label)
            results(model, whole_predict_disp, whole_actual_disp, label, whole_min_max, whole_resid, whole_res_min_max, "validation2", "regress")
            
        else: #regression
            evaluate_validation_regression_model(gs_val_tree, pen_val_tree, INDCORP_val_tree, NAICS_val_tree, selected)
        validation = False

        input("         Press Enter to continue to feature selection...")
   
    exit()
    if predict_method == "classify":
        feat_based_train_acc = []
        feat_based_test_acc = []
        feat_based_train_f1 = []
        feat_based_test_f1 = []
        
    else: #regression
        feat_based_train_MSE = []
        feat_based_test_MSE = []
        feat_based_train_Rsq = []
        feat_based_test_Rsq = []
    
    for feature_index in range(0, len(input_column_names), 1):
        test_feature = input_column_names[feature_index]
        test_input_column_names = input_column_names.copy()

        with open('Model_'+ str(len(input_column_names))+'_summary.txt', 'a') as f:
            with redirect_stdout(f):
                print("==================================================================")
                print("Removing: " + str(test_feature) + ", Index: " + str(feature_index+1), ", Features left: " + str(len(input_column_names)))
 
        print("==================================================================")
        print("Removing: " + str(test_feature) + ", Index: " + str(feature_index+1), ", Features left: " + str(len(input_column_names)))
 
        if len(input_column_names) != 1:
            test_input_column_names.remove(test_input_column_names[feature_index])
            if predict_method == "classify":
                X_feature_test = np.delete(X_train, feature_index, 1)
            else:
                gs_tree_test.above_X = np.delete(gs_tree.above_X, feature_index, 1)
                gs_tree_test.below_X = np.delete(gs_tree.below_X, feature_index, 1)
                
                gs_tree_test.above_X_df = pd.DataFrame(gs_tree_test.above_X, index=gs_tree_test.ID_above, columns=test_input_column_names)                
                gs_tree_test.below_X_df = pd.DataFrame(gs_tree_test.below_X, index=gs_tree_test.ID_below, columns=test_input_column_names)
                
                pen_tree_test.above_X = np.delete(pen_tree.above_X, feature_index, 1)
                pen_tree_test.below_X = np.delete(pen_tree.below_X, feature_index, 1)

                pen_tree_test.above_X_df = pd.DataFrame(pen_tree_test.above_X, index=pen_tree_test.ID_above, columns=test_input_column_names)                
                pen_tree_test.below_X_df = pd.DataFrame(pen_tree_test.below_X, index=pen_tree_test.ID_below, columns=test_input_column_names)

                NAICS_tree_test.group1_X = np.delete(NAICS_tree.group1_X, feature_index, 1)
                NAICS_tree_test.group2_X = np.delete(NAICS_tree.group2_X, feature_index, 1)
                NAICS_tree_test.group3_X = np.delete(NAICS_tree.group3_X, feature_index, 1)

                NAICS_tree_test.group1_X_df = pd.DataFrame(NAICS_tree_test.group1_X, index=NAICS_tree_test.ID_group1, columns=test_input_column_names) 
                NAICS_tree_test.group2_X_df = pd.DataFrame(NAICS_tree_test.group2_X, index=NAICS_tree_test.ID_group2, columns=test_input_column_names) 
                NAICS_tree_test.group3_X_df = pd.DataFrame(NAICS_tree_test.group3_X, index=NAICS_tree_test.ID_group3, columns=test_input_column_names) 

                INDCORP_tree_test.group1_X = np.delete(INDCORP_tree.group1_X, feature_index, 1)
                INDCORP_tree_test.group2_X = np.delete(INDCORP_tree.group2_X, feature_index, 1)
                INDCORP_tree_test.group3_X = np.delete(INDCORP_tree.group3_X, feature_index, 1)
                
                INDCORP_tree_test.group1_X_df = pd.DataFrame(INDCORP_tree_test.group1_X, index=INDCORP_tree_test.ID_group1, columns=test_input_column_names) 
                INDCORP_tree_test.group2_X_df = pd.DataFrame(INDCORP_tree_test.group2_X, index=INDCORP_tree_test.ID_group2, columns=test_input_column_names) 
                INDCORP_tree_test.group3_X_df = pd.DataFrame(INDCORP_tree_test.group3_X, index=INDCORP_tree_test.ID_group3, columns=test_input_column_names) 
                
        else: #if last feature, just keep and run
            if predict_method == "classify":
                X_feature_test = X_train 
                        
        if predict_method == "classify":
            X_df = pd.DataFrame(data=X_feature_test, columns=test_input_column_names, index=ID_train)
            y_df = pd.DataFrame(data=y_train, columns=output_column_names, index=ID_train)
        
            chosen_params = {'hidden_layers': report.data.iloc[best_model_index]['hidden_layers'],
                              'activation': report.data.iloc[best_model_index]['activation'],
                              'first_neuron': report.data.iloc[best_model_index]['first_neuron'],
                              'optimizer': report.data.iloc[best_model_index]['optimizer'],
                              'shapes': report.data.iloc[best_model_index]['shapes'],
                              'dropout': report.data.iloc[best_model_index]['dropout'],
                              'epochs': report.data.iloc[best_model_index]['epochs'],
                              'batch_size': report.data.iloc[best_model_index]['batch_size'],
                              'input_neurons': X_feature_test.shape[1],
                              'kernel_l2': report.data.iloc[best_model_index]['kernel_l2'],
                              'activity_l2': report.data.iloc[best_model_index]['activity_l2'],
                              'lr': report.data.iloc[best_model_index]['lr']}
            
            kf_idx_train_list, kf_idx_test_list = cv_ID(X_df, y_df, n_cv, predict_method)
            train_acc_cv, test_acc_cv, train_f1_cv, test_f1_cv = cross_validation(X_feature_test, y_train, kf_idx_train_list, kf_idx_test_list, chosen_params, False, predict_method)
    
            feat_based_train_acc.append(np.mean(train_acc_cv))
            feat_based_test_acc.append(np.mean(test_acc_cv))
            feat_based_train_f1.append(np.mean(train_f1_cv))
            feat_based_test_f1.append(np.mean(test_f1_cv))
        
        else: #regression        
            train_MSE_cv, test_MSE_cv, train_Rsq_cv, test_Rsq_cv = regression_data_partition(gs_tree_test, pen_tree_test, INDCORP_tree_test, NAICS_tree_test, n_cv, selected, 0, chosen_params)
            
            feat_based_train_MSE.append(np.mean(train_MSE_cv))
            feat_based_test_MSE.append(np.mean(test_MSE_cv))
            feat_based_train_Rsq.append(np.mean(train_Rsq_cv))
            feat_based_test_Rsq.append(np.mean(test_Rsq_cv))
        
    with open('Model_'+ str(len(input_column_names))+'_summary.txt', 'a') as f: 
        with redirect_stdout(f):
            print("\n==================================================================\n")
            if predict_method == "classify":
                print("Training-Highs (Acc): " + str(feat_based_train_acc))
                print("Testing-Highs (Acc): " + str(feat_based_test_acc))
                print("Training-Highs (F1): " + str(feat_based_train_f1))
                print("Testing-Highs (F1): " + str(feat_based_test_f1))
            else:
                print("Training-Highs (MSE): " + str(feat_based_train_MSE))
                print("Testing-Highs (MSE): " + str(feat_based_test_MSE))
                print("Training-Highs (R²): " + str(feat_based_train_Rsq))
                print("Testing-Highs (R²): " + str(feat_based_test_Rsq))                

    if predict_method == "classify":
        least_impact_feature_index_train = feat_based_train_f1.index(max(feat_based_train_f1))
        least_impact_feature_index_test = feat_based_test_f1.index(max(feat_based_test_f1))
    
        ranked_feats.append(input_column_names[least_impact_feature_index_test])
        ranked_feat_accs.append(max(feat_based_test_acc))
        ranked_feat_f1.append(max(feat_based_test_f1))
    else:
        least_impact_feature_index_train = feat_based_train_Rsq.index(max(feat_based_train_Rsq))
        least_impact_feature_index_test = feat_based_test_Rsq.index(max(feat_based_test_Rsq))
        
        
        ranked_feats.append(input_column_names[least_impact_feature_index_test])
        ranked_feat_MSE.append(min(feat_based_test_MSE[least_impact_feature_index_test], feat_based_train_MSE[least_impact_feature_index_test]))
        ranked_feat_Rsq.append(min(feat_based_test_Rsq[least_impact_feature_index_test], feat_based_train_Rsq[least_impact_feature_index_test]))     
    
    with open('Model_'+ str(len(input_column_names))+'_summary.txt', 'a') as f: 
        with redirect_stdout(f):          
            print("\n==================================================================\n")
            print("DROPPING " + str(input_column_names[least_impact_feature_index_test]) + " at Index " + str(least_impact_feature_index_test+1))
            if predict_method == "classify":
                print("Accuracies List: " + str(ranked_feat_MSE))
                print("F1 List: " + str(ranked_feat_f1))
            else:
                print("Accuracies List: " + str(ranked_feat_accs))
                print("F1 List: " + str(ranked_feat_Rsq))             
            print("Ranked Features (from least to most important): " + str(ranked_feats)) #from most important to least important
            print("\n==================================================================\n")

    print("\n==================================================================\n")
    print("DROPPING " + str(input_column_names[least_impact_feature_index_test]) + " at Index " + str(least_impact_feature_index_test+1))
    if predict_method == "classify":
        print("Accuracies List: " + str(ranked_feat_accs))
        print("F1 List: " + str(ranked_feat_f1))   
    else:
        print("MSE List: " + str(ranked_feat_MSE))
        print("R² List: " + str(ranked_feat_Rsq))
    print("Ranked Features (from least to most important): " + str(ranked_feats)) #from most important to least important
    print("\n==================================================================\n")

    if predict_method == "classify":
        X_train = np.delete(X_train, least_impact_feature_index_test, 1)
    else:
        gs_tree.below_X = np.delete(gs_tree.below_X, least_impact_feature_index_test, 1)
        gs_tree.above_X = np.delete(gs_tree.above_X, least_impact_feature_index_test, 1)
        pen_tree.below_X = np.delete(pen_tree.below_X, least_impact_feature_index_test, 1)
        pen_tree.above_X = np.delete(pen_tree.above_X, least_impact_feature_index_test, 1)
        NAICS_tree.group1_X = np.delete(NAICS_tree.group1_X, least_impact_feature_index_test, 1)
        NAICS_tree.group2_X = np.delete(NAICS_tree.group2_X, least_impact_feature_index_test, 1)
        NAICS_tree.group3_X = np.delete(NAICS_tree.group3_X, least_impact_feature_index_test, 1)
        INDCORP_tree.group1_X = np.delete(INDCORP_tree.group1_X, least_impact_feature_index_test, 1)
        INDCORP_tree.group2_X = np.delete(INDCORP_tree.group2_X, least_impact_feature_index_test, 1)
        INDCORP_tree.group3_X = np.delete(INDCORP_tree.group3_X, least_impact_feature_index_test, 1)
        
    input_column_names.remove(input_column_names[least_impact_feature_index_test])
    
ranked_feats.reverse()
if predict_method == "classify":
    ranked_feat_accs.reverse()
    ranked_feat_f1.reverse()
else:
    ranked_feat_MSE.reverse()
    ranked_feat_Rsq.reverse()

with open('Summary_One_NN.txt', 'w') as f: 
    with redirect_stdout(f):
        print("Ranked Features (from most to least important): " + str(ranked_feats)) 
        if predict_method == "classify":
            print("Accuracies List (from most to least important): " + str(ranked_feat_accs))
            print("F1 List (from most to least important): " + str(ranked_feat_f1))
        else:
            print("MSE List (from most to least important): " + str(ranked_feat_MSE))
            print("R² List (from most to least important): " + str(ranked_feat_Rsq))            

rf = ranked_feats
if predict_method == "classify":
    metric_1 = ranked_feat_accs #Accuracy
    metric_2 = ranked_feat_f1 #F1
else:
    metric_1 = ranked_feat_MSE #Accuracy
    metric_2 = ranked_feat_Rsq #F1  

rfe_fig = plt.figure(figsize=(7, 5))   
plt.scatter(rf, metric_2)
plt.plot(rf, metric_2)
plt.xticks(rotation=90, horizontalalignment='right', fontsize='x-small')
plt.gcf().subplots_adjust(bottom=0.40)
plt.xlabel('Incrementally Removed Features (most to least important)')
ax = rfe_fig.add_subplot()
if predict_method == "classify":
    plt.title("No. of audited returns  = 2,387")
    plt.ylabel('F1 Score')
    plt.ylim(0.4, 0.6)
    ax.yaxis.set_ticks([0.4, 0.5, 0.6])
else:
    plt.title("No. of audited returns  = 877")
    plt.ylabel('R-squared')
    plt.ylim(0, 0.4)
    ax.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
plt.show()




