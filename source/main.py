#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:40:56 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd

# =============================================================================
# data pipeline  from scratch
# =============================================================================


import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import GlobalsFunctions

#'''retrive data'''
#from classes.DataDownloader import DataDownloader
#dd = DataDownloader()