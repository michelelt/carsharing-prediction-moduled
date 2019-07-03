#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:38:10 2019

@author: mc
"""
import pandas as pd
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')
from GlobalsFunctions import crs_, df2gdf

import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from GlobalsFunctions import *
import io, pkgutil    
import geoplot 
from matplotlib import colors   


data_path = '../../data/'
city='Vancouver'
train = pd.read_csv(data_path+city+'/Regression/train_emer.csv').fillna(0).sort_values('FID').set_index('FID')

columns_to_remove =[
'geometry',
'MAPID',
'geometry_nwf',
'geometry_neigh',
        ]

train = train.drop(columns_to_remove, axis=1)
train.to_csv(data_path+city+'/Regression/dataset_MRMR.csv')
zzz =pd.read_csv(data_path+city+'/Regression/dataset_MRMR.csv')
