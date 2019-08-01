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
import pytz
from GlobalsFunctions import haversine, crs_
import geoplot


# =============================================================================
# data pipeline  from scratch
# =============================================================================
from importlib import import_module

merge_neigh_with_census = getattr(
        import_module('03_vancouver_opendata_merger'), 
        'merge_neigh_with_census'
        )

upload_data = getattr(
        import_module('03_vancouver_opendata_merger'), 
        'upload_data'
        )

merge_tiles_and_building_info = getattr(
        import_module('03_vancouver_opendata_merger'), 
        'merge_tiles_and_building_info'
        )

add_emergencies_column = getattr(
        import_module('04_b_merge_streets_311'), 
        'add_emergencies_column'
        )



from classes.DataPreprocesser import DataPreprocesser
from classes.DataDownloader   import DataDownloader
from classes.TilesMapCreator  import TilesMapCreator
from classes.MetricCreator    import MetricCreator
from classes.LocalMemoryChecker import LocalMemoryChecker
import datetime


# =============================================================================
# service_functions
# =============================================================================




if __name__ == '__main__':
    city = 'Vancouver'
    data_path = '../../data/'
    
    i_date = datetime.datetime(2017, 10, 1,  0,  0,  0)
    f_date = datetime.datetime(2017, 10, 31, 23, 59, 59)
    
    
 
  
    dp = DataPreprocesser(city, data_path, i_date=i_date, f_date=f_date)
    dp.upload_bookigns()
    dp.standard_filtering()
    
    


    data_path_neigh = data_path+city+'/Opendata/'
    
    neigh = upload_data(data_path+city+'/Opendata/')
    building_info = gpd.read_file(data_path+city+'/Opendata/'+\
                                  'zoning_districts_shp/zoning_districts.shp')\
            .to_crs(crs_)
            
    neigh_with_features = merge_neigh_with_census(neigh, 
                                                  data_path+city+'/Opendata/')
    neigh_with_features = neigh_with_features\
                        .reset_index()\
                        .reset_index()\
                        .rename(columns={'index':'FID'})
    
    neigh = merge_tiles_and_building_info(neigh_with_features, building_info)
                        

    mc = MetricCreator(dp.booking, neigh, i_date, f_date, data_path)
    tiles = mc.tiles
    mc.merge_tiles_with_bookings()
    
    neigh_with_metric = mc.compute_metrics_per_tile(save=True)
    
    add_emergencies_column = getattr(
        import_module('04_b_merge_streets_311'), 
        'add_emergencies_column'
        )

    dataset = add_emergencies_column(neigh_with_metric)
    dataset.iloc[0:20].to_csv('../../data/Vancouver/Regression/dataset_train_emer.csv')
    dataset.iloc[20:22].to_csv('../../data/Vancouver/Regression/dataset_test_emer.csv')
#    
    
#    # =========================================================================
#    # test on tiles
#    # =========================================================================
#   
#    i_date = datetime.datetime(2017, 10, 1,  0,  0,  0)
#    f_date = datetime.datetime(2017, 10, 31, 23, 59, 59)
#    
#    dp2 = DataPreprocesser(city, data_path, i_date=i_date, f_date=f_date)
#    dp2.upload_bookigns()
#    dp2.standard_filtering()
#    dp2.filter_weekends(filter_friday=True)
#    
#
#    tmc = TilesMapCreator(dp2.booking, i_date, f_date, data_path)
#    tmc.create_empity_tiles_map(500, 0.001, save=True)
#    zzz = tmc.tiles
#   
#    mc2 = MetricCreator(dp2.booking, tmc.tiles, i_date, f_date, data_path)
#    mc2.merge_tiles_with_bookings()
#    tiles_with_metric = mc2.compute_metrics_per_tile(save=True)
#    
#    '''
#    add here neigh 24 x 24  merging with census data
#    '''
        
    


        

        

    
#    
#    
    



