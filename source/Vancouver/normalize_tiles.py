#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:01:00 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd
import numpy as np

import os, sys
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname('..'))
from GlobalsFunctions import starts_labels, finals_labels

def normalize_dataset(data_path, city, ind_variable='time'):
    train = pd.read_csv(data_path+city+'/Regression/dataset_train_emer.csv').fillna(0)
    test  = pd.read_csv(data_path+city+'/Regression/dataset_test_emer.csv')
    
    
    
    
    # =============================================================================
    # Detect which FID1 tiles are not in FID2
    # =============================================================================
    fid_train_not_in_fid_test = []
    for fid in train.FID:
        if fid not in list(test.FID): 
            fid_train_not_in_fid_test.append(fid)
    
    # =============================================================================
    # Detect which FID2 tiles are not in FID1
    # =============================================================================      
    fid_test_not_in_fid_train = []
    for fid in train.FID:
        if fid not in list(train.FID): 
            fid_test_not_in_fid_train.append(fid)
            
            
    
    # =============================================================================
    # makes both dataset equal number of tiles
    # =============================================================================
    train = train.set_index('FID').drop(fid_train_not_in_fid_test).reset_index()
    test  = test.set_index('FID').drop(fid_test_not_in_fid_train).reset_index()
    
    MYLABEL = 'c_start_0'
    if sum(train[MYLABEL] - test[MYLABEL]) == 0: 
        print('THE TWO DATAFRAMES ARE THE SAME!')
        
    
    if ind_variable == 'space':
        init_df = train.copy()
        for i in range(0,7):
            init_df[starts_labels[i]] = init_df[starts_labels[i]] + test[starts_labels[i]]
            init_df[finals_labels[i]] = init_df[finals_labels[i]] + test[finals_labels[i]]
        
#        train = init_df.sample(n=int(0.7*len(init_df)))
#        test  = init_df.loc[np.setdiff1d(init_df.index, train.index)]
        
        
        
    index2FID_train = train['FID']
    index2FID_test  = test['FID']
    
    return train, test, index2FID_train, index2FID_test, 1, 2

    
    
    columns_to_delete=[
#    'MAPID', 'lat', 'lon', 'geometry_nwf', 'geometry_neigh','geometry',
#    'FID' 
    ]
    
    
    # =============================================================================
    # Normalization of the data
    # =============================================================================
    train_norm = train.drop(columns_to_delete, axis=1)
    test_mean = train.mean()
    test_std  = train.std()
    train_norm = (train_norm - train_norm.mean())/train_norm.std()
    

    test_norm = test.drop(columns_to_delete, axis=1)
    test_norm = (test_norm - test_mean)/test_std
    
    return train_norm, test_norm, index2FID_train, index2FID_test, test_mean, test_mean


#
#data_path = './../../data/'
#city = 'Vancouver'
#train, test, index2FID_train, index2FID_test  = normalize_dataset(data_path, city, ind_variable='space')













