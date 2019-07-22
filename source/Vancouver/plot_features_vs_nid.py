#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:56:56 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

data_path = '../../data/Vancouver/'
#data = 'dataset_MRMR_1.csv'
data = 'Regression/dataset_emer.csv'
df = pd.read_csv(data_path+data)

col_1 = df.columns[  0: 25].tolist()
col_2 = df.columns[ 25: 50].tolist()
col_3 = df.columns[ 50: 75].tolist()
col_4 = df.columns[100:124].tolist()

col_banches =  [col_1, col_2, col_3,  col_4]

    

#for banch  in col_banches:
#    fig,ax = plt.subplots(5,5, figsize=(50,50))
#    for i in range(len(banch)):
#        col = i%5
#        row = (int(i/5)) %5 
#
#        
#        ax[row,col].plot(df.index, df[banch[i]])
#        ax[row,col].set_ylabel(banch[i])
#        ax[row,col].set_xticks([])
target  =  df['c_start_0']  
mapid = df[['MAPID', 'NAME']]


for c in df.columns:
    if 'm_age' in c or\
       'f_age' in c or\
       'c_start' in c or\
       'c_final' in c or\
       'count' in c or\
       'geometry' in c or\
       'FID' in c or\
       'MAPID' in c or\
       'NAME' in c:
        df=df.drop(c, axis=1)
        
df['c_start_0'] = target
df[['MAPID', 'NAME']]  = mapid
df.to_csv(data_path+'dataset_1_target.csv', index=False)
zzz = pd.read_csv(data_path+'dataset_1_target.csv')

#df_1 = df.iloc[0:20]
#df_2 = df.iloc[20:22]
#
#
#from sklearn import preprocessing
#
#scaler = preprocessing.StandardScaler()
#scaler = scaler.fit(df_1.values)
#train = pd.DataFrame(index=df_1.index,
#                    columns=df_1.columns,
#                    data=scaler.transform(df_1.values)
#                    )
#
#valid = pd.DataFrame(index=df_2.index,
#                    columns=df_2.columns,
#                    data=scaler.transform(df_2.values)
#                    )
#    

