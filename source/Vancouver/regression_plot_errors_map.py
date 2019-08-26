#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:26:20 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd
import decimal

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from GlobalsFunctions import create_errors_df, get_best_config, df_coords2gdf
import json

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np

#def compute_operative_area(data_path, city):

    
    


city = 'Vancouver'
data_path  = './../../data/'

filename =  'Vancouver_filtered_binned_2017-10-01T00-00-00_2017-10-31T23-59-59.csv'
    
df = pd.read_csv(data_path+city+'/'+filename, nrows=None)
limits = pd.read_csv(data_path+city+'/Vancouver_limits.csv')
df = df[ (df.start_lat >= limits.min_lat.values[0])\
        &(df.end_lat >= limits.min_lat.values[0])\
        &(df.start_lon >= limits.min_lon.values[0])\
        &(df.end_lon >= limits.min_lon.values[0])\
        &(df.start_lat <= limits.max_lat.values[0])\
        &(df.end_lat <= limits.max_lat.values[0])\
        &(df.start_lon <= limits.max_lon.values[0])\
        &(df.end_lon <= limits.max_lon.values[0])
]
gdf = df_coords2gdf(df, df.start_lat, df.start_lon)


neighs = gpd.read_file(data_path\
                       +city\
                       +'/Opendata/Vancouver_macroArea/Vancouver_macroArea.shp')\
                       .to_crs({'init': 'epsg:4326'})[['MAPID', 'geometry']]#ne
                       

    
                       
res_rfr = data_path+city+'/Regression/output_rfr/rfr_regression_dist.csv'
res_svr = data_path+city+'/Regression/output_svr/svr_regression_dist.csv'
#
#
#df = pd.read_csv(res_svr)
#
errors_df = create_errors_df(res_rfr, res_svr)
best_sol = get_best_config(errors_df)
#
svr_start = best_sol['svr']['start']
errors_data = errors_df[ (errors_df['kernel'] == 'linear')\
                        &(errors_df['is_normed'] == False)\
                        &(errors_df['target'].str.contains('start')) 
                    ].iloc[0]['err']


e_dict = json.loads(errors_data, parse_float=float)
e_df = pd.Series(e_dict).reset_index()
e_df['index'] = e_df['index'].astype(int)
e_df = e_df.set_index('index')
neighs['errors'] = e_df


fig,ax = plt.subplots()
neighs.plot(column='errors', ax=ax)
gdf.plot(ax=ax, markersize=0.5, color='red', alpha=0.5)



