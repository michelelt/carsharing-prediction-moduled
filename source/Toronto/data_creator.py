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
    city = 'Toronto'
    data_path = './../../data/'
    
#    i_date = datetime.datetime(2017, 10, 25, 0, 0, 0)
#    f_date = datetime.datetime(2017, 10, 31, 23, 59, 59)
 
#    i_date = datetime.datetime(2017, 10, 25,  0, 0,  0)
#    f_date = datetime.datetime(2017, 10, 31, 23,59, 59)
    
    i_date = None
    f_date = None
    
    dp = DataPreprocesser(city, data_path, i_date=i_date, f_date=f_date)
    dp.upload_bookigns()
    dp.standard_filtering()

    tmc = TilesMapCreator(dp.booking, i_date, f_date, data_path)
    tmc.create_empity_tiles_map(500, 0.001, save=True)
   
  
    mc = MetricCreator(dp.booking, tmc.tiles, i_date, f_date, data_path)
    mc.merge_tiles_with_bookings()
    mc.compute_metrics_per_tile(save=True)
    


        

        

    
#    
#    
    



