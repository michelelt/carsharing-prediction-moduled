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
#    i_date = datetime.datetime(2016, 9, 6, 0, 0, 0)
#    f_date = datetime.datetime(2019, 9, 15, 0, 0, 0)
## =============================================================================
## 
## =============================================================================
    dp = DataPreprocesser(city, data_path, i_date=i_date, f_date=f_date)
    dp.upload_bookigns()
    dp.standard_filtering()

#    dp.detect_spatial_outliers()

   
    tmc = TilesMapCreator(dp.booking, data_path)
#    
    tmc.create_empity_tiles_map(500, 0.001, save=True)
#    tiles = tmc.tiles
#   
#   
    mc = MetricCreator(dp.booking, tmc.tiles, data_path)
    mc.merge_tiles_with_bookings()
    tiles_with_metrics2 =  mc.compute_metrics_per_tile(save=False)
    tiles_with_metrics2.plot()
##    from vancouver_opendata_merger import neigh_with_features
#    
#    to_norm =  tiles_with_metrics2[['count_start', 'count_end', 'geometry']]
#    to_norm['sum'] =  to_norm['count_start']  + to_norm['count_start']
#    to_norm['sum_norm']  = (to_norm['sum'] - to_norm['sum'].mean())/to_norm['sum'].std()
#    to_norm.plot(column='sum_norm')
#    

#import os
#os.remove('../data/Vancouver/Vancouver_filtered_binned.csv')
    

    
    
    
    
    
    

        

    
#    
#    
    



