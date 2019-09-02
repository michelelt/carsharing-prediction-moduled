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
 
    


#feature_ranks = pd.read_csv(data_path+city+'/Regression/feature_ranks.csv')
#most_ranked = feature_ranks.mean().sort_values(ascending=False)
#df = reg.complete_dataset
def run_rfr(reg):
    ## =============================================================================
    ## RFR regression   
    ## =============================================================================
    start = time.time()
    for target in sorted(reg.targets):
        for n_estimators in range(10, 101, 10):
            for train_index, valid_index in  loo.split(reg.complete_dataset):
                
                print('RFR', target, n_estimators, valid_index)
    
                reg.set_norm(False)
                reg.split_datasets(target, train_index, valid_index)
                reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
                res.append(reg.results)
                
                reg.set_norm(True)
                reg.split_datasets(target, train_index, valid_index)
    
                reg.perform_regression('rfr', n_estimators=n_estimators, random_state=0)
                res.append(reg.results)
                print()
                
    
    
    end = time.time() - start
    print('RFR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
    df = pd.DataFrame(res)
    df.to_csv(data_path+city+'/Regression/output_rfr/rfr_regression_dist.csv')



def run_svr(reg):
    # =============================================================================
    # SVR regression
    # =============================================================================
    res = []
    start = time.time()
    for target in sorted(reg.targets):
        for kernel in sorted(['rbf', 'linear', 'poly']):
            for train_index, valid_index in loo.split(reg.complete_dataset):
                
                print('SVR', target, kernel, valid_index)
    
    
                reg.set_norm(False)
                reg.split_datasets(target, train_index, valid_index)
                reg.perform_regression('svr',  kernel=kernel)
                res.append(reg.results)
                
                reg.set_norm(True)
                reg.split_datasets(target, train_index, valid_index)
                reg.perform_regression('svr', kernel=kernel)
                res.append(reg.results)
                print()
                
                
    end = time.time() - start
    print('SVR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
    df = pd.DataFrame(res)
    df.to_csv(data_path+city+'/Regression/output_svr/svr_regression_dist.csv')



if __name__=='__main__':
    
    city = 'Vancouver'
    data_path  = './../../data/'
    
    loo = LeaveOneOut()
    res = []
    #
    start = time.time()
    reg = Regression(data_path, city, norm=True)
    reg.add_distance_as_feature(base_in_downtown=True)
    reg.preprocess_data()
    df = reg.targets_df
    means = df.median().to_frame().sort_index().T
#    run_rfr(reg)
#    run_svr(reg)