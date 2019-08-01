#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:26:20 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from GlobalsFunctions import create_errors_df, get_best_config
import json

import matplotlib.pyplot as plt


city = 'Vancouver'
data_path  = './../../data/'

neighs = gpd.read_file(data_path\
                       +city\
                       +'/Opendata/Vancouver_macroArea/Vancouver_macroArea.shp')\
                       .to_crs({'init': 'epsg:4326'})[['MAPID', 'geometry']]
                       
from sklearn.model_selection import LeaveOneOut 
import time
import datetime
import pandas as pd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from classes.Regression import Regression
 
    
city = 'Vancouver'
data_path  = './../../data/'

loo = LeaveOneOut()
res = []
#
start = time.time()
reg = Regression(data_path, city, norm=True)

df = reg.complete_dataset

reg_mapid = df.MAPID.tolist()
df_mapid = neighs.MAPID.tolist()

for df_mid in df_mapid:
    if df_mid not in reg_mapid:
        print (df_mapid)
        
#fig, ax = plt.subplots()
#neighs.plot(ax=ax)
#neighs[neighs.MAPID == 'GW'].plot(ax=ax, color='red')
                       
                       
#res_rfr = data_path+city+'/Regression/output_rfr/rfr_regression_dist.csv'
#res_svr = data_path+city+'/Regression/output_svr/svr_regression_dist.csv'
#
#
#df = pd.read_csv(res_svr)
#
#errors_df = create_errors_df(res_rfr, res_svr)
##best_sol = get_best_config(errors_df)
#
##svr_start = best_sol['svr']['start']
#errors_data = errors_df[ (errors_df['kernel'] == 'linear')\
#                        &(errors_df['is_normed'] == False)\
#                        &(errors_df['target'].str.contains('start')) 
#                    ].iloc[0]['err']
#
#error_dict = json.loads(errors_data)
