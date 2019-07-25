#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:14:47 2019

@author: mc
"""

from sklearn.model_selection import LeaveOneOut 
import time
import datetime
import pandas as pd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from classes.Regression import Regression
 
    
city = 'Vancouver'
data_path  = './../../data/'

loo = LeaveOneOut()
res = []
#
start = time.time()
reg = Regression(data_path, city, norm=True)
reg.preprocess_data()

feature_ranks = pd.read_csv(data_path+city+'/Regression/feature_ranks.csv')
most_ranked = feature_ranks.mean().sort_values(ascending=False)

# =============================================================================
# RFR regression   
# =============================================================================
start = time.time()
for target in ['c_start_0']:
    for n_estimators in range(10, 101, 100):
        for train_index, valid_index in  loo.split(reg.complete_dataset):
            for nof in range(1, 10):
            
                print('RFR', target, n_estimators, valid_index, nof)
                reg.set_norm(False)
                reg.split_datasets(target, train_index, valid_index, 
                                   features_to_keep=most_ranked.iloc[0:nof].index.tolist())
    #            train, valid = reg.train, reg.valid
                train_target, valid_target  = reg.train_target, reg.valid_target
                reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
                res.append(reg.results)
                
                reg.set_norm(True)
                reg.split_datasets(target, train_index, valid_index, 
                                   features_to_keep=most_ranked.iloc[0:nof].index.tolist())
    #            train, valid = reg.train, reg.valid
                train_target, valid_target  = reg.train_target, reg.valid_target
    
                reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
                res.append(reg.results)
                print()
            


end = time.time() - start
print('RFR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
df = pd.DataFrame(res)
#df.to_csv(data_path+city+'/Regression/output_rfr/rfr_regression_fs.csv')





## =============================================================================
## SVR regression
## =============================================================================
#start = time.time()
#for target in sorted(reg.targets):
#     for kernel in ['rbf', 'linear', 'poly']:
#        for train_index, valid_index in loo.split(reg.complete_dataset):
#            
#            print('SVR', target, kernel, valid_index)
#
#            reg.set_norm(False)
#            reg.split_datasets(target, train_index, valid_index)
#            train_target, valid_target  = reg.train_target, reg.valid_target
#            reg.perform_regression('svr',  kernel  = kernel)
#            res.append(reg.results)
#            
#            reg.set_norm(True)
#            reg.split_datasets(target, train_index, valid_index)
#            train_target, valid_target  = reg.train_target, reg.valid_target
#    
#            reg.perform_regression('svr', kernel=kernel)
#            res.append(reg.results)
#            print()
#            
#            
#end = time.time() - start
#print('SVR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
#df = pd.DataFrame(res)
#df.to_csv(data_path+city+'/Regression/output_svr/svr_regression.csv')
#            