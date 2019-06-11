#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:51:44 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from GlobalsFunctions import *

import time

data_path = './../data/Vancouver/Opendata/'
def upload_data(data_path):
    
    #complete dataset with 673 columns, remove all, keeping 2
    neigh = gpd\
            .read_file(data_path+'Vancouver_macroArea/Vancouver_macroArea.shp')\
            .reset_index()[['MAPID', 'geometry']]\
            .set_index('MAPID')
            
    #neigh dataset with neighbour names        
    neigh_names = pd\
                    .read_csv(data_path+'Vancouver_macroArea.csv')\
                    .dropna()\
                    .set_index('MAPID')\
                    .rename(columns={'Unnamed: 0': 'NAME'})
    
    #attach to first dataset the neigh names            
    neigh = neigh.join(neigh_names['NAME']).reset_index().set_index('NAME')

    return neigh


def columns_id_to_keep(data_path):
    file = open(data_path + 'support_data/' + 'columns_to_keep.txt','r')
    lines = file.readlines()
    file.close()
    columns_id_to_keep = []
    for line in lines:
        line = line.split('-')
        if len(line) == 1:
            columns_id_to_keep.append(int(line[0]))
            
        else:
            for el in range(int(line[0]), int(line[1])+1):
                columns_id_to_keep.append(el)
    columns_id_to_keep_array = np.array(columns_id_to_keep) -1
    return columns_id_to_keep_array

def column_labels_to_keep(data_path):
    file = open(data_path + 'support_data/' + 'columns_label_to_keep.txt','r')
    lines = file.readlines()
    
    lines2 = []
    for line in lines: lines2.append(line.rstrip().lstrip())
    
    return lines2


def merge_neigh_with_census(neigh, data_path):

    # =========================================================================
    # census csv parsing
    # =========================================================================
    mapping = pd.read_csv(data_path+'census/map.csv')
    do_not_consider = [
            13,14,15,16,17,18,19,20,21,22,23,
            26,27,28,29,30,31,32,33,34,40,41,
            46,47,48,50,51,56,57,58,59,60,61,
            62,63,65,65,66,67,68,69,70,71,72,
            73,74,75,76,77,78,79,80,81,83,84,
            85,86,87,90,92,93,94,95,96,53,88,
            64,91, 1, 2, 3,12]
    
    for file_id in mapping.id:
        if file_id in do_not_consider: continue
        if file_id < 10: file_id_s = '0'+str(file_id)
        else: file_id_s = str(file_id)
        
        df = pd.read_csv(data_path+'census/%s.csv'%file_id_s )
        df = df.set_index('Variable')
        df = df[df.columns[1:len(df.columns)]].T
        try:
            neigh = neigh.join(df)
        except ValueError:
            print('Exception:', file_id_s)
    
    neigh_with_feat = neigh[column_labels_to_keep(data_path)]
    return neigh_with_feat



def merge_tiles_and_building_info(tiles, building_info):

    categories = building_info.groupby('CATEGORY').count()
    test_join = gpd.sjoin(tiles, building_info)
    test_join = test_join.groupby(['FID','CATEGORY']).size() 
    census_mc = gpd.GeoDataFrame(index=tiles['FID'].astype(int),
                                crs={'init': 'epsg:4326'}, 
                                columns=list(categories.index))
    
    
    for index in test_join.index:
        zone_id = index[0], 
        category = index[1]
        value = test_join.loc[(zone_id, category)][0]
        census_mc.loc[int(zone_id[0])][category] = value
    census_mc = census_mc.fillna(0)
#    
    census_mc = census_mc.join(tiles.set_index('FID')).reset_index()
    return census_mc


    
def merge_squares_and_neighs(toll_list, neigh, tiles):
    dict_key_square = {}
    for toll in toll_list:
        area = pd.DataFrame()
        init_time = time.time()
        
        neigh = neigh.reset_index()
        tiles = tiles.set_index('FID')
        print ("toll", toll)
        
        for neigh_id in range(0,len(neigh.index)):
            print(neigh_id)
            a=0
            one_neigh = neigh.iloc[neigh_id+a:neigh_id+a+1]
            new_area = gpd.sjoin(one_neigh, tiles, how='left', op='intersects')

            tiles_to_check = tiles.loc[new_area.index_right.tolist()]
            tiles_to_check['MAPID'] = neigh.iloc[neigh_id+a:neigh_id+a+1]['MAPID'].values[0]
            
            for fid in tiles_to_check.index:
                square = gpd.GeoDataFrame(tiles_to_check.loc[fid], 
                                          crs={'init': 'epsg:4326'}).T
                intersect = gpd.overlay(one_neigh, square, how='intersection')
                
                if neigh_id not in dict_key_square.keys():
                    dict_key_square[neigh_id] = []
                else:
                    dict_key_square[neigh_id].append(fid)
        
                a1 = tiles.loc[fid].geometry.area
                a2 = intersect['geometry'].area.values[0]
                
                if a2/a1 <= toll:
                    tiles_to_check = tiles_to_check.drop(fid, axis=0)
                 
            
            area = area.append(tiles_to_check)
                    
        neigh = neigh.set_index('MAPID')
        tiles = tiles.reset_index() 
    return area

'''
# =============================================================================
# implement the merging with the bookings!!!!
# =============================================================================
'''

neigh= upload_data(data_path)
neigh_with_features= merge_neigh_with_census(upload_data(data_path),data_path)
building_info = gpd\
        .read_file(data_path+'zoning_districts_shp/zoning_districts.shp')\
        .to_crs(crs_)
tiles = gpd\
        .read_file(data_path+'../Vancouver_tiles_metric/Vancouver_tiles_metric.shp')

tiles =  merge_tiles_and_building_info(tiles, building_info)

squares_overlapped = merge_squares_and_neighs([0.51], neigh, tiles) 
zzz = squares_overlapped.set_index('MAPID')\
                    .join(neigh_with_features.set_index('MAPID'),
#                          how='left',
#                          on='MAPID',
                          lsuffix = '_nwf').reset_index()
#

#tiles.plot()

