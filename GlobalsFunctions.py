#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:19:46 2019

@author: mc
"""

import geopandas as gpd
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import scale
from math import *
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np



def create_colors_dict(index_min,index_max):
    NUM_COLORS = (index_max + 1 - index_min)
#    colors_dict = {}
#    cm = plt.get_cmap('gist_rainbow')
#    for i in range(index_min, index_max+1):
#        colors_dict[i] = cm(1. * (i+index_max+1) / NUM_COLORS)
#    return colors_dict
    all_colors = [k for k,v in pltc.cnames.items()]
    all_colors_dict = {}
    for i in range(0, NUM_COLORS):
        all_colors_dict[i-(-index_min)] = all_colors[i]
#        print(i-index_max, all_colors[i])
    return all_colors_dict
#create_colors_dict(-15,15)

crs_ = {'init': 'epsg:4326'}

time_bins = [ '1:00 - 6:59',
             '7:00 - 9:59',
            '10:00 - 12:59',
            '13:00 - 15:59',
            '16:00 - 18:59',
            '19:00 - 21:59',
            '22:00 - 0:59'
            ]


paths_dict={
    'data_path'               : '../../data/',
    'init_dataset'            : 'vancouver_bookings_augmented.csv',
    'tiles'                   : 'tiles/tiles.shp',
    'analysis_dataset'        : 'Vancouver_bookings_augmented_reduced_columns_hs.csv',
    'analysis_dataset_mid'    : 'Vancouver_bookings_augmented_reduced_columns_hs_mid.csv',
    'final_dataset_tiles'     : 'VanocuverHM_timebins_mon_fri_shp/VanocuverHM_timebins_mon_fri.shp',
    'neighbours'              : 'Vancouver_macroArea/Vancouver_macroArea.shp',
    'neigh_names'             : 'Vancouver_macroArea.csv',
    'census_columns'          : 'census/column_names.txt',
    'censuns_mapper'          : 'census/map.csv',
    'vancouver_HM_Gi'         : 'Vancouver_HM_Gi/Vancouver_HM_Gi.shp',
    'support_data_path'       : './support_data/',
    'columns_id_label'        : 'columns_id_label.txt',
    'neigh_with_feat_renamed' : 'neigh_with_feat_renamed/neigh_with_feat_renamed.shp',
    'zoning_districts'        : 'zoning_districts_shp/zoning_districts.shp',
    'building_info_and_Gi'    : 'building_info_and_Gi/building_info_and_Gi.shp',
    'building_info_and_Gi_and_census' : 'building_info_and_Gi_and_census.csv',
    'building_info_Gi_census_bin' : 'building_info_Gi_census_bin/building_info_Gi_census_bin.shp'
}

##colors = {
## -14 : (1.0, 0.0, 0.16, 1.0),
## -13 : (1.0, 0.007419183889772136, 0.0, 1.0),
## -12 : (1.0, 0.19819819819819823, 0.0, 1.0),
## -11 : (1.0, 0.36777954425013254, 0.0, 1.0),
## -10 : (1.0, 0.5585585585585586, 0.0, 1.0),
## -9 : (1.0, 0.7281399046104929, 0.0, 1.0),
## -8 : (1.0, 0.918918918918919, 0.0, 1.0),
## -7 : (0.9114997350291467, 1.0, 0.0, 1.0),
## -6 : (0.7207207207207207, 1.0, 0.0, 1.0),
## -5 : (0.5511393746687864, 1.0, 0.0, 1.0),
## -4 : (0.36036036036036034, 1.0, 0.0, 1.0),
## -3 : (0.19077901430842603, 1.0, 0.0, 1.0),
## -2 : (0.0, 1.0, 0.0, 1.0),
## -1 : (0.0, 1.0, 0.16866961838498848, 1.0),
##  0 : (0.0, 1.0, 0.3584229390681005, 1.0),
##  1 : (0.0, 1.0, 0.5481762597512125, 1.0),
##  2 : (0.0, 1.0, 0.716845878136201, 1.0),
##  3 : (0.0, 1.0, 0.9065991988193131, 1.0),
##  4 : (0.0, 0.9239130434782604, 1.0, 1.0),
##  5 : (0.0, 0.7320971867007668, 1.0, 1.0),
##  6 : (0.0, 0.5615942028985503, 1.0, 1.0),
##  7 : (0.0, 0.36977834612105687, 1.0, 1.0),
##  8 : (0.0, 0.19927536231884035, 1.0, 1.0),
##  9 : (0.0, 0.007459505541347444, 1.0, 1.0),
## 10 : (0.16304347826086973, 0.0, 1.0, 1.0),
## 11 : (0.35485933503836337, 0.0, 1.0, 1.0),
## 12 : (0.5253623188405799, 0.0, 1.0, 1.0),
## 13 : (0.7171781756180736, 0.0, 1.0, 1.0),
## 14 : (0.8876811594202902, 0.0, 1.0, 1.0),
## 15 : (1.0, 0.0, 0.9205029838022163, 1.0)
#        
#        }

# =============================================================================
# convert from truncated column label to complete one
# =============================================================================
def compose_mising_lab(label, df):
    gender = label[0]
    _id = label[9]
#    print('---',_id, label)
    return gender + '_' + df.loc[int(_id)-1, 'name']

def replace_truncated_labels(neigh):
    columns = open('../../data/census/column_names.txt', 'r')
    column_names = columns.readlines()[0].split(';')
    columns.close()
    ad_hoc_metrics = ['count_start','count_end', 'c_start_0', 'c_final_0', 
                      'c_start_1', 'c_final_1', 'c_start_2', 'c_final_2', 
                      'c_start_3', 'c_final_3', 'c_start_4', 'c_final_4', 
                      'c_start_5', 'c_final_5', 'c_start_6', 'c_final_6']
    
    t_leaving = pd.read_csv(paths_dict['support_data_path'] + 'x_leaving_d.csv')
    t_commut = pd.read_csv(paths_dict['support_data_path'] + 'x_commute_d.csv')
    
    column_names_short = neigh.columns
    dict_names = {}
    for c_short in column_names_short:
        if c_short in ad_hoc_metrics: continue
        if '_leavin_' in c_short: 
            dict_names[c_short] = compose_mising_lab(c_short, t_leaving)
            
        if '_commut_' in c_short: 
            dict_names[c_short] = compose_mising_lab(c_short, t_commut)
    #        print (dict_names[c_short])
    #        continue
        for c_name in column_names:
            if c_name[0:10] == c_short:
                
                if c_short in dict_names.keys(): 
                    print ('Already inserted')
                else:
                    dict_names[c_short] = c_name
                    break
                
    neigh = neigh.rename(columns=dict_names)
    return neigh
'''
# =============================================================================
# set of columns of censuns file to save for analysis
# =============================================================================
def columns_id_to_keep():
    file = open(paths_dict['support_data_path'] + 'columns_to_keep.txt','r')
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

def column_labels_to_keep():
    file = open(paths_dict['support_data_path'] + 'columns_label_to_keep.txt','r')
    lines = file.readlines()
    
    lines2 = []
    for line in lines: lines2.append(line.rstrip().lstrip())
    
    return lines2
'''

# =============================================================================
# upload from text file the columns on which I want to compute the most
# correlated features
# =============================================================================
def upload_labels_for_corr():
    file = open(support_data_path+'labels_for_corr.txt', 'r')
    columns = file.readlines()
    
    for i in range(0, len(columns)):
        columns[i] = columns[i].strip()
    return columns


# =============================================================================
# Clustering data with KMeans, considering one computed par on bookings,
# removing the other computed par. on bookings 
# keeping all the other cenusus parameters
# =============================================================================

def KMeans_clustering(neigh, label_to_keep, n_cluster):
    labels_for_corr = upload_labels_for_corr() #computed_analysis
    def labels_to_not_consider_in_cls(label_to_keep, labels_for_corr):
        list_out = []
        for label in labels_for_corr:
            if label != label_to_keep:
                list_out.append(label)
        return list_out
    
    
    
    km5 = cluster.KMeans(n_clusters=n_cluster)
    l = labels_to_not_consider_in_cls(label_to_keep,labels_for_corr)
    l.append('MAPID')
    l.append('MacroZone')
    l.append('geometry')
    l.append('NAME')
    neigh_cls = neigh.drop(l,axis=1)
    neigh_cls = neigh_cls.astype(float)
    km5cls = km5.fit(neigh_cls.values)
    neigh['cl'] = km5cls.labels_ 
    return  neigh


# =============================================================================
# remove dollar symbol from labels
# =============================================================================
def remove_dollar_symbol(columns):
    out_list = []
    for c in columns:
        if '$' in c:
            c = c.replace('$', 'S')
        out_list.append(c)
    return out_list


# =============================================================================
# upload and rename columns
# =============================================================================
def upload_neighbours():
    neigh = gpd.read_file(paths_dict['data_path'] + paths_dict['neigh_with_feat_renamed'] )
    
    columns_dict ={}
    f = open(paths_dict['support_data_path'] + paths_dict['columns_id_label'])
    lines = f.readlines()
    f.close()
    
    for line in lines:
        line = line.split(',')
        if line[0] != 'geometry':
            columns_dict[line[0]] = line[1].lstrip().rstrip()
        
    neigh = neigh.rename(columns=columns_dict)
    return neigh, columns_dict



# =============================================================================
# classic haversine
# =============================================================================
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
#    return in meter

    return int(km*1000)

def df2gdf(df, lat, lon):
    geometry = [Point(xy) for xy in zip(lon, lat)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    return gdf

def str2polygon(geometry):
    multiareas = str(geometry)
    multiareas = multiareas.replace("POLYGON","")\
            .replace('), (', ';')\
            .split(';')
    geo_points=[]   
    for ma in multiareas:
        ma = ma.replace('((', '').replace('))', '')
        ma = ma.split(',')
        
        for point in ma:
            A = (float(point.strip().split(' ')[0]) , 
                 float(point.strip().split(' ')[1])
                     )
            geo_points.append(A)           

    #    print (ps)
    return Polygon(geo_points)



def upload_bookings(path, paths_dict, *args, **kwargs ):
    if args == 0:
        df = pd.read_csv(path + paths_dict['init_dataset'])
    else: 
        df = pd.read_csv(path + paths_dict['init_dataset'], nrows=kwargs.get('nrows'))
    
    return df



