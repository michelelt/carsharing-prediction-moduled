#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:19:46 2019

@author: mc
"""

import geopandas as gpd
import pandas as pd
import numpy as np


from paramiko import SSHClient
from scp import SCPClient

from shapely.geometry import Point, Polygon, MultiPolygon



crs_ = {'init': 'epsg:4326'}

time_bins = [ '1:00 - 6:59',
             '7:00 - 9:59',
            '10:00 - 12:59',
            '13:00 - 15:59',
            '16:00 - 18:59',
            '19:00 - 21:59',
            '22:00 - 0:59'
            ]

starts_labels = ['c_start_%d'%tb for tb in range(0,7)]
finals_labels = ['c_final_%d'%tb for tb in range(0,7)]

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

def df_coords2gdf(df, lat, lon):
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


def ssh_connection():
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('bigdatadb.polito.it',
                username='cocca',
                password=open('../../credential/bigdatadb.txt','r').readline(),
                look_for_keys=False)
    return ssh


def download_file(ssh, src, dst):
    with SCPClient(ssh.get_transport()) as scp:
        
#        if not os.path.isdir('../../MicheleRankings/outputs/'):
#            os.mkdir('../../MicheleRankings/outputs/')
        
        scp.get(remote_path=src, 
                 local_path=dst, 
                 recursive=True)
        
        print(src + ' Completede downloaded and saved in ' + dst)
        
def compute_target_labels():
    starts  = []
    finals  = [] 
    for i in range(0,7): 
    	starts.append('c_start_%d'%i) 
    	finals.append('c_final_%d'%i)
            
    targets_dict = {'starts':starts, 'finals':finals}      
    return targets_dict
    
    

def compute_mean_err_perc(train, target):
    return sum(abs(train-target)/target)/(len(target))



