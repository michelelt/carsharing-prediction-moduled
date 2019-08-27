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
        
        print(src + ' Completed downloaded and saved in ' + dst)
        
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

def compute_rmse(train, target):
    square = np.power(train-target, 2)
    square = square.div(len(target))
    return np.sqrt(sum(square))



def get_best_config(errors_df):
    
    def booking_type(x) : return x['target'][2:-2]
    errors_df['booking_type'] = errors_df.apply(booking_type, axis=1)


    best_svr = errors_df[(errors_df.reg_type=='svr')]
    best_svr_g = best_svr\
                .groupby(['kernel', 'is_normed', 'booking_type'])\
                .mean()\
                .sort_values('err_mean_perc')
            
    best_rfr = errors_df[(errors_df.reg_type=='rfr')]
    print(best_rfr.columns)
    best_rfr_g = best_rfr.groupby(['n_estim', 'is_normed', 'booking_type'])\
                .mean()\
                .sort_values('err_mean_perc')
    
    
    
    min_start_found = False
    min_final_found = False
    sol = {}
    for reg_type in ['svr', 'rfr']:
        if reg_type == 'svr': df = best_svr_g
        else: df = best_rfr_g
        for index, row in df.iterrows():
        
            if index[2] == 'final' and not min_final_found:
                best_min_final = {
                        'variable': index[0],
                        'normed': index[1],
                        'targets': index[2]
                        }
                min_final_found = True
                
            if index[2] == 'start' and not min_start_found:
                best_min_start = {
                        'variable': index[0],
                        'normed': index[1],
                        'targets': index[2]
                        }
                min_start_found = True
                
            if min_start_found and min_final_found: 
                sol[reg_type] = {'start': best_min_start, 'final':best_min_final}
                min_start_found = False
                min_final_found = False
                break
        
    return sol


def create_errors_df(res_rfr, res_svr):

    rfr_df =  pd.read_csv(res_rfr).drop('rank', axis=1)
    svr_df = pd.read_csv(res_svr)
    errors_list = []
    
    for is_normed in rfr_df.is_normed.unique():
        for n_estim in rfr_df.n_estimators.unique():
            for target in rfr_df.target.unique():
                for nof in rfr_df.nof.unique():
        #            normed = True
        #            target = 'c_start_0'
        #            n_estim = 40
                    
                    a = rfr_df[rfr_df.is_normed == is_normed]
                    a = a[a.target==target]
                    a = a[a.n_estimators == n_estim]
                    a = a[a.nof == nof]


                    a = a.set_index('FID_valid')
                    if is_normed:
                        y_pred_label = 'rb_y_pred'
                        y_valid_label = 'rb_y_valid'
                    else:
                        y_pred_label = 'y_pred_valid'
                        y_valid_label = 'y_valid'            
                        
                        
                    err_abs =  abs((a[y_pred_label] - a[y_valid_label])).div(a[y_valid_label])
                    err_perc = err_abs*100
                    rmse = compute_rmse(a[y_pred_label], a[y_valid_label])
                    errors_list.append(
                            { 'is_normed':is_normed,
                              'target':target,
                              'n_estim':n_estim,
                              'err_mean_perc': err_perc.mean(),
                              'err_median_perc':err_perc.median(),
                              'err_mean_abs':err_abs.mean(),
                              'err_median_abs':err_abs.median(),
                              'rmse':rmse,
                              'err': err_perc.to_json(),
                              'reg_type': 'rfr',
                              'nof':nof
                            })
                    
                    
    
    for is_normed in svr_df.is_normed.unique():
        for kernel in svr_df.kernel.unique():
            for target in svr_df.target.unique():
                for nof in svr_df.nof.unique():
                    
                    a = svr_df[svr_df.is_normed == is_normed]
                    a = a[a.kernel == kernel] 
                    a = a[a.target == target]
                    a = a[a.nof == nof]
                    
                    if is_normed:
                        y_pred_label = 'rb_y_pred'
                        y_valid_label = 'rb_y_valid'
                    else:
                        y_pred_label = 'y_pred_valid'
                        y_valid_label = 'y_valid'            
                        
                        
                    err_abs = abs((a.set_index('FID_valid')[y_pred_label] - 
                                    a.set_index('FID_valid')[y_valid_label]))\
                                    .div(a.set_index('FID_valid')[y_valid_label])   
                    err_perc = err_abs*100
                    rmse = compute_rmse(a.set_index('FID_valid')[y_pred_label], 
                                        a.set_index('FID_valid')[y_valid_label])
                    errors_list.append(
                            { 'is_normed':is_normed,
                              'target':target,
                              'kernel':kernel,
                              'err_mean_perc': err_perc.mean(),
                              'err_median_perc':err_perc.median(),
                              'err_mean_abs':err_abs.mean(),
                              'err_median_abs':err_abs.median(),
                              'rmse':rmse,
                              'err': err_perc.to_json(),
                              'reg_type': 'svr',
                              'nof':nof
                            })

#                    errors_list.append(
#                            { 'is_normed':is_normed,
#                              'target':target,
#                              'kernel':kernel,
#                              'err_mean': err_perc.mean(),
#                              'err_median':err_perc.median(),
#                              'err': err_perc.to_json(),
#                              'reg_type': 'svr',
#                              'nof':nof
#                            })
            
    errors_df = pd.DataFrame(errors_list)
    return errors_df



