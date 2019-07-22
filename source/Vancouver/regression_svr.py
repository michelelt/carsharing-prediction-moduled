#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:25 2019

@author: mc
"""

import pandas as pd
from sklearn.svm import SVR



# =============================================================================
# import datasets
# =============================================================================
city = 'Vancouver'
data_path  = './../../data/'
train = pd.read_csv('%s%s/Regression/dataset_train_emer.csv'%(data_path, city))
valid = pd.read_csv('%s%s/Regression/dataset_test_emer.csv'%(data_path, city))
complete_dataset = train.append(valid, ignore_index=True)

# =============================================================================
# SVR models oparams
# =============================================================================
kernel_labels = ['RBF', 'Linear', 'Polynomial']
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
svrs = [svr_rbf, svr_lin, svr_poly]
metrics = pd.DataFrame(
        columns=['FID_valid', 'y_pred_valid', 'y_valid', 'er_r_pred_train', 'kernel']
        )

norm = False
targets = []
for c in complete_dataset.columns:
    if 'c_start' in c or 'c_final' in c:
        targets.append(c)
        
        

s =  pd.Series(index=metrics.columns)
for target in targets:
    print(target)
    for i in range(len(complete_dataset)):
        valid = complete_dataset.loc[i].to_frame().T
        train = complete_dataset.loc[~complete_dataset.index.isin([i])]
        
        mean, std  = train.mean(), train.std()

        
        

        # =============================================================================
        # preprocess data
        # =============================================================================
        train_target = train[target]
        valid_target = valid[target]
        for c in train.columns:
            if ('sum' in c) or ('count' in  c)\
            or ('start' in c) or ('final'  in c)\
            or ('Gi_' in c) or ('m_age' in c)\
            or ('f_age' in c) or ('NAME' in  c)\
            or ('MAPID' in c) or ('FID' in c)\
            or ('geometry' in c):
                train = train.drop(c, axis=1)
                valid = valid.drop(c,axis=1)

        if norm==True:
            #reubuilt to norm
            train[target] = train_target
            valid[target] = valid_target
            
            #norm
            mean, std  = train.mean(), train.std()
            train = (train-mean)/std
            valid = (valid-mean)/std
            
            #resplit
            train_target = train[target]
            valid_target = valid[target]

        
        # =================================================================
        # SVR regression
        # =============================================================================
    
        for ix, svr in enumerate(svrs):
            regressor = svr.fit(train, train_target)
            y_pred_train = svr.predict(train)
            er_r_pred_train =  sum(abs(y_pred_train-train_target)/train_target)/(len(train_target))
            y_pred_valid = svr.predict(valid)
            
            s['FID_valid'] = i
            s['y_pred_valid'] =  y_pred_valid[0]
            s['y_valid'] = valid_target.values[0]
            s['er_r_pred_train'] = er_r_pred_train
            s['kernel']  = kernel_labels[ix]
            s['target']  = target
            s['mean_target'] =  mean[target]
            s['std_target'] = std[target]
            s['is_normed'] = norm

            metrics = metrics.append(s, ignore_index=True)
 

metrics.to_csv('%s%s/Regression/output_svr/metrics_svr_norm_%s.csv'%(data_path, city, norm))           
 

