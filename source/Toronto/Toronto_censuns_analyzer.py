#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:21:59 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd

data_path = '../../data/'
city      = 'Toronto'
file      = '2016_neighbourhood_profiles.csv'
df = pd.read_csv(data_path+city+'/Opendata/'+file, encoding='unicode_escape')
df = df.T


'''
Quartieri  11 features
'''
abs_path =  '/Users/mc/Desktop/OpendataAlreadySeen_1/gcc/Projects/Open Data/Files/Data Upload - May 2010/May2010_WGS84/icitw_wgs84.shp'
file2  = 'neighbourhoods_planning_areas_wgs84/NEIGHBORHOODS_WGS84.shp'
gdf = gpd.read_file(data_path+city+'/Opendata/'+file2)

#tor = gdf[gdf.CMANAME=='Toronto']
#gdf.plot(edgecolor='red')

#
#subneighs = [el.upper() for el in df.columns]
#for col in gdf.NAME.unique().tolist():
#    col = col[:-5].upper()
#    if col not in subneighs:
#        print(col)


