#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:12:35 2019

@author: mc
"""

import time
import datetime
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from classes.Regression import Regression
 
    
city = 'Vancouver'
data_path  = './../../data/'

neighs = gpd.read_file(data_path\
               +city\
               +'/Opendata/Vancouver_macroArea/Vancouver_macroArea.shp')\
               .to_crs({'init': 'epsg:4326'})[['MAPID', 'geometry']]
               
centroids = neighs.geometry.centroid

               
train = pd.read_csv('%s%s/Regression/dataset_train_emer.csv'%(data_path, city))
valid = pd.read_csv('%s%s/Regression/dataset_test_emer.csv'%(data_path, city))
data = train.append(valid, ignore_index=True).set_index('MAPID')

neighs.set_index('MAPID', inplace=True)
neighs['NAME'] = data['NAME']
neighs['data'] = data['count_start'] + data['count_end']
neighs.reset_index(inplace=True)

        

gdf = neighs
gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
gdf['coords'] = [coords[0] for coords in gdf['coords']]

'''
Macro-zoning
'''
s = pd.Series([2,0,2,0,1,
               1,1,1,1,1,
               1,1,0,1,0,
               2,2,2,2,2,
               0,2,2,1,2
               ])
gdf['MacroZone'] = s

'''
Plot Macro zone with labeled Neighborhood
'''
fig,ax = plt.subplots(1,1, figsize=(20,20))
gdf.plot(column='data', cmap='GnBu', edgecolor='black',ax=ax, label=gdf['NAME'])
title = ""
for idx, row in gdf.iterrows():
    plt.annotate(s=str(idx)+'-'+row['MAPID']+'\n',
                 xy=row['coords'],
                 horizontalalignment='center',
                 fontsize=8
                 )
    plt.annotate(s=row['NAME'], xy=row['coords'],
             horizontalalignment='center', 
             fontsize=5
             )
    if idx % 5 == 0 and idx!=0: div = '\n'
    else : div = '; '
    title  +=str(idx) +'-'+ row['MAPID']+'-'+row['NAME'] + div
ax.set_title(title)
#plt.savefig('../map_negh.pdf', bbox_inchemas = 'tight', format='pdf')

#gdf[['MAPID', 'NAME', 'geometry', 'MacroZone']].to_file("Vancouver_macroArea.shp")
#zzz = gpd.read_file("Vancouver_macroArea.shp")
         
#
