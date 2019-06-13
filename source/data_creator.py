#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:40:56 2019

@author: mc
"""

#__all__ = ["DataDownloader"]

import pandas as pd
import geopandas as gpd

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



if __name__ == '__main__':
    city = 'Vancouver'
    data_path = './../data/'
#    
## =============================================================================
##     try to implement to save date in the file name
## =============================================================================
    i_date = datetime.datetime(2017, 9, 6, 0, 0, 0)
    f_date = datetime.datetime(2017, 9, 26, 1, 0, 0)
## =============================================================================
## 
## =============================================================================
    dp = DataPreprocesser(city, data_path, i_date=i_date, f_date=f_date)
    dp.upload_bookigns()
    dp.standard_filtering()
    

   
    tmc = TilesMapCreator(dp.booking, i_date, f_date, data_path)
    tmc.create_empity_tiles_map(500, 0.001, save=True)
    tiles = tmc.tiles
   
  
    mc = MetricCreator(dp.booking, tmc.tiles, i_date, f_date, data_path)
#    bookings = mc.df
    mc.merge_tiles_with_bookings()
    mc.compute_metrics_per_tile(save=True)
    
#    if city == 'Vancouver':
#        from vancouver_opendata_merger import squares_overlapped
##        
#    columns_to_delete=[
#    'MAPID', 'FID', 'lat', 'lon', 'geometry_nwf', 'geometry_neigh'
#            ]
#    corr_df = squares_overlapped.drop(columns_to_delete, axis=1)
#    corr_df = corr_df.astype(float)
#    corr = corr_df.corr()
#    
#    row_to_keep=[]
#    for c in corr.columns:
#        if ('sum' in c)   or ('final' in  c)\
#        or ('start' in c) or ('count' in c) \
#        or ('Gi_' in c) :
#            row_to_keep.append(c)
#            
#    corr = corr.loc[row_to_keep].T.drop(row_to_keep)
#    
#    
#    from sklearn.linear_model import LinearRegression
#    
#    for tb in range(0,7):
#        prediction_label = 'c_start_%d'%tb
#        y = squares_overlapped[prediction_label].values
#        
#        df_temp = corr[prediction_label].sort_values(ascending=False).iloc[0:20]
#        X = squares_overlapped[df_temp.index].astype(float).values
#        
#        reg = LinearRegression().fit(X,y)
#        print('tb: %d, R^2 = %f' % (tb, reg.score(X,y)))
    
    



        

        

    
#    
#    
    



