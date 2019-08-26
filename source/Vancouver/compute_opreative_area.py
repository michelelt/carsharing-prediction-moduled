#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:50:20 2019

@author: mc
"""
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import geopandas as gpd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  
from GlobalsFunctions import df_coords2gdf



city = 'Vancouver'
data_path  = './../../data/'

#filename =  'Vancouver_filtered_binned_2017-10-01T00-00-00_2017-10-31T23-59-59.csv'
#    
#df = pd.read_csv(data_path+city+'/'+filename, nrows=1000)
#limits = pd.read_csv(data_path+city+'/Vancouver_limits.csv')
#
#neighs = gpd.read_file(data_path\
#                       +city\
#                       +'/Opendata/Vancouver_macroArea/Vancouver_macroArea.shp')\
#                       .to_crs({'init': 'epsg:4326'})[['MAPID', 'geometry']]#ne
#
#df = df[ (df.start_lat >= limits.min_lat.values[0])\
#        &(df.end_lat >= limits.min_lat.values[0])\
#        &(df.start_lon >= limits.min_lon.values[0])\
#        &(df.end_lon >= limits.min_lon.values[0])\
#        &(df.start_lat <= limits.max_lat.values[0])\
#        &(df.end_lat <= limits.max_lat.values[0])\
#        &(df.start_lon <= limits.max_lon.values[0])\
#        &(df.end_lon <= limits.max_lon.values[0])
#]
#
#gdf = df_coords2gdf(df, df.start_lat, df.start_lon)
#my_polygon = gdf.geometry.bounds
#
#my_polygon.plot()

#fig, ax = plt.subplots()
#neighs.plot(ax=ax)
#points =  pd.DataFrame(list(zip(df.start_lon, df.start_lat))).values
#hull = ConvexHull(points)
#ax.plot(points[:,0], points[:,1], 'o')
#for simplex in hull.simplices:
#     ax.plot(points[simplex, 0], points[simplex, 1], 'k-')