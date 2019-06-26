#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:58:58 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd
import time

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import  matplotlib.pyplot as plt


# =============================================================================
# data pipeline  from scratch
# =============================================================================
from classes.DataPreprocesser import DataPreprocesser
from classes.DataDownloader   import DataDownloader
from classes.TilesMapCreator  import TilesMapCreator
from classes.MetricCreator    import MetricCreator
from classes.LocalMemoryChecker import LocalMemoryChecker
import datetime

city = 'Vancouver'
data_path = './../data/'

def datetime2ts(dt):
    return int(time.mktime(dt.timetuple()))



i_date_train = datetime.datetime(2017, 10, 1, 0, 0, 0)
i_ts_train = datetime2ts(i_date_train)

f_date_train = datetime.datetime(2017, 10, 24, 23, 59, 59)
f_ts_train = datetime2ts(f_date_train)


 
i_date_test = datetime.datetime(2017, 10, 25,  0, 0,  0)
i_ts_test = datetime2ts(i_date_test)

f_date_test = datetime.datetime(2017, 10, 31, 23,59, 59)
f_ts_test = datetime2ts(f_date_test)


business_day  =  [0,1,2,3,4]
original_dataset = pd\
    .read_csv(data_path+city+'/Vancouver_filtered_binned_merged_None_none.csv')
original_dataset.loc[:,'dayofweek'] = pd.DatetimeIndex(original_dataset.init_date).dayofweek
    
    
tiles = gpd.read_file(data_path+city+'/Vancouver_tiles/Vancouver_tiles.shp')
 


df_train = original_dataset[
             (original_dataset.init_time  >= i_ts_train)
            &(original_dataset.final_time <= f_ts_train)
            &(original_dataset.dayofweek.isin(business_day))
        ]

df_test = original_dataset[
             (original_dataset.init_time  >= i_ts_test)
            &(original_dataset.final_time <= f_ts_test)
            &(original_dataset.dayofweek.isin(business_day))
        ]


mc_train = MetricCreator(df_train, tiles, i_date_train, f_date_train, data_path)
mc_train.compute_metrics_per_tile(save=False)
#train = mc_train.tiles


mc_test = MetricCreator(df_test, tiles, i_date_test, f_date_test, data_path)
mc_test.compute_metrics_per_tile(save=False)
#test = mc_test.tiles
#
#
if city == 'Vancouver':
    from vancouver_opendata_merger import create_squares_overlapped
    
    

    
train = create_squares_overlapped(
        i_date_train,
        f_date_train,
        mc_train.tiles,
        city,
        data_path+city+'/Opendata/'
        )[3]

test = create_squares_overlapped(
        i_date_test,
        f_date_test,
        mc_test.tiles,
        city,
        data_path+city+'/Opendata/'
        )[3]


'''
# =============================================================================
# Normalize data from neighboirhoud
# =============================================================================
'''
tiles_per_neigh  = train.groupby('MAPID').count()['FID']
train = train.set_index('MAPID')
test = test.set_index('MAPID')
columns_not_norm = [ 
'participation_rate', 'employment_rate', 'unemployment_rate', 'geometry_neigh', 'geometry_nwf',
'Gi_0', 'Gi_1', 'Gi_2', 'Gi_3', 'Gi_4', 'Gi_5', 'Gi_6',
'FID', 'Commercial', 'Comprehensive Development', 'Historical Area',
'Industrial', 'Light Industrial', 'Limited Agriculture', 'Multi-Family Dwelling',
'One-Family Dwelling', 'Other',  'Two-Family Dwelling', 
'count_start', 'count_end',  'c_start_1', 'c_final_1', 'c_start_2', 'c_final_2',
'c_start_3', 'c_final_3', 'c_start_4', 'c_final_4', 'c_start_5', 'c_final_5',
'c_start_6', 'c_final_6', 'c_start_0', 'c_final_0', 'lat', 'lon',
'sum_1', 'sum_2', 'sum_3', 'sum_4', 'sum_5', 'sum_6', 'sum_0']


for index, row in tiles_per_neigh.iteritems():
    train.loc[index,train.columns.difference(columns_not_norm)] =\
        train.loc[index,train.columns.difference(columns_not_norm)]\
        .astype(float)\
        .div(row)
        
    test.loc[index,test.columns.difference(columns_not_norm)] =\
        test.loc[index,test.columns.difference(columns_not_norm)]\
        .astype(float)\
        .div(row)
        
 
      

train = train.reset_index()
train.to_csv(data_path+city+'/Regression/train.csv', index=False)
from merge_streets_311 import add_emergencies_column
add_emergencies_column()


test = test.reset_index()
test.to_csv(data_path+city+'/Regression/test.csv', index=False)



        
        
        
        





