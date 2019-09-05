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
from GlobalsFunctions import get_best_config,\
    create_errors_df,\
    compute_target_labels,\
    compute_feature_rank
    
    
import sys
 

def run_rfr(reg, best_sol, most_ranked):
	## =============================================================================
	## RFR regression   
	## =============================================================================
    res=[]
    loo = LeaveOneOut()
    start = time.time()
    rfr = best_sol['rfr']
    for key in rfr.keys():
        var = rfr[key]['variable']
        normed = rfr[key]['normed']
        targets = compute_target_labels()[key+'s']
        
        for target in targets:
            for train_index, valid_index in loo.split(reg.complete_dataset):
                for nof in range(1,len(most_ranked)+1):
	            
    	            print('RFR - FS', target, var, valid_index, nof)
    
    	            reg.set_norm(normed)
    	            reg.split_datasets(target, train_index, valid_index,
                                    features_to_keep=most_ranked.iloc[0:nof].index)
    	            reg.perform_regression('rfr', n_estimators=int(var), random_state=0)
    	            res.append(reg.results)
    	           
    	            print()
	            
    end = time.time() - start
    print('RFR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
    df = pd.DataFrame(res)
    df.to_csv(data_path+city+'/Regression/output_rfr/rfr_regression_dist_fs.csv')
    

def run_svr(reg, best_sol, most_ranked):
	# =============================================================================
	# SVR regression
	# =============================================================================
    res = []
    loo = LeaveOneOut()

    start = time.time()
    svr = best_sol['svr']
    for key in svr.keys():
        var = svr[key]['variable']
        normed = svr[key]['normed']
        targets = compute_target_labels()[key+'s']
        
        for target in targets:
            for train_index, valid_index in loo.split(reg.complete_dataset):
                for nof in range(1,len(most_ranked)+1):
	            
                    print('SVR - FS', normed, target, var, valid_index, nof)
                    
                    reg.set_norm(True)
                    reg.split_datasets(target, train_index, valid_index,
                       features_to_keep=most_ranked.iloc[0:nof].index)
                    
                    reg.perform_regression('svr', kernel=var, random_state=0, nu=0.1)
                    res.append(reg.results)
    	           
                    print()
	            
	            
    end = time.time() - start
    print('SVR Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
    df = pd.DataFrame(res)
    df.to_csv(data_path+city+'/Regression/output_svr/svr_regression_dist_fs.csv')
    



if __name__=='__main__':
    city = 'Vancouver'
    data_path  = './../../data/'
    res_rfr = data_path+city+'/Regression/output_rfr/rfr_regression_dist.csv'
    res_svr = data_path+city+'/Regression/output_svr/svr_regression_dist.csv'

    
#    feature_ranks = pd.read_csv(data_path+city+'/Regression/feature_ranks.csv')
    feature_ranks = compute_feature_rank(res_rfr, False)
    most_ranked = feature_ranks.mean().sort_values(ascending=False)
#    
##    
    errors_df = create_errors_df(res_rfr, res_svr)
    best_sol = get_best_config(errors_df)
##    
    start = time.time()
##    
    reg = Regression(data_path, city, norm=True)
    reg.preprocess_data()
##    
    reg.add_distance_as_feature(base_in_downtown=True)
    reg.add_area_as_feature('km2')
    reg.normalize_features_per_area()
    reg.normalize_targets_per_area()

    


    
    if sys.argv[1].lower() == 'rfr':
        run_rfr(reg, best_sol, most_ranked)
    elif sys.argv[1].lower() == 'svr':
        run_svr(reg, best_sol, most_ranked)
    else:
        print('error')
       
    


#            