#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:14:47 2019

@author: mc
"""

from sklearn.model_selection import LeaveOneOut 
import time
import datetime

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

    
start = time.time()
for target in sorted(reg.targets):
    for n_estimators in range(10, 101, 10):
        for train_index, valid_index in  loo.split(reg.complete_dataset):
            
            print(target, n_estimators, valid_index)
            reg.set_norm(False)
            reg.split_datasets(target, train_index, valid_index)
#            train, valid = reg.train, reg.valid
            train_target, valid_target  = reg.train_target, reg.valid_target
            reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
            res.append(reg.results)
            
            reg.set_norm(True)
            reg.split_datasets(target, train_index, valid_index)
#            train, valid = reg.train, reg.valid
            train_target, valid_target  = reg.train_target, reg.valid_target

            reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
            res.append(reg.results)
            print()
            
            
            
end = time.time() - start
print('Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
#            