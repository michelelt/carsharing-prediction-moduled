#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:42:01 2019

@author: mc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import numpy as np

import os, sys
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname('../..'))
#from GlobalsFunctions import starts_labels, finals_labels

def normalize_dataset(data_path, city, ind_variable='time'):
    train = pd.read_csv(data_path+city+'/Regression/dataset_train_emer.csv').fillna(0)
    test  = pd.read_csv(data_path+city+'/Regression/dataset_test_emer.csv')

#    
    MYLABEL = 'c_start_0'
    if sum(train[MYLABEL] - test[MYLABEL]) == 0: 
        print('THE TWO DATAFRAMES ARE THE SAME!')
        
        
        
    index2FID_train = train['FID']
    index2FID_test  = test['FID']

    columns_to_delete=[
    'MAPID', 'FID', 'NAME', 'MAPID','geometry',
    ]
    
    train_norm = train.drop(columns_to_delete, axis=1)
    test_norm = test.drop(columns_to_delete, axis=1)
    
    
    # =============================================================================
    # Normalization of the data
    # =============================================================================
    test_mean = train_norm.mean()
    test_std  = train_norm.std()
    train_norm = (train_norm - train_norm.mean())/train_norm.std()
    

    test_norm = (test_norm - test_mean)/test_std
    
    return train_norm, test_norm, index2FID_train, index2FID_test, test_mean, test_mean


#
#data_path = './../../data/'
#city = 'Vancouver'
#train, test, index2FID_train, index2FID_test  = normalize_dataset(data_path, city, ind_variable='space')













