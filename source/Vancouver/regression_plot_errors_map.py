#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:26:20 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd
import decimal
from math import sqrt

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from GlobalsFunctions import create_errors_df, get_best_config, df_coords2gdf, crs_
import json

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np

#def compute_operative_area(data_path, city):

def compute_index_plot(pics, i):
    n = str(bin(i)).replace('0b', '')
    for zeros_to_add in range(1, int(sqrt(pics)) -len(n)+1):
        n = '0'+n
        
    r = n[0]
    c = n[1]
    
    return int(r),int(c)


def compute_errors_per_FID_per_target(errors_df, best_sol):
    epz_pt = pd.DataFrame()
    for reg in best_sol.keys():
        for t_type in best_sol[reg]:
            single_best_sol = best_sol[reg][t_type]
    
    #        svr_start = best_sol['svr']['start']
            if reg == 'svr': var ='kernel'
            else: var =  'n_estim' 
            errors_data = errors_df[ (errors_df[var] == single_best_sol['variable'])\
                                    &(errors_df['is_normed'] == single_best_sol['normed'])\
                                    &(errors_df['target'].str.contains(t_type))\
                                    &(errors_df['nof'] == single_best_sol['nof'])
                                ]
            
            
            
            for sol in range(0,7):
                e_dict = json.loads(errors_data.iloc[sol]['err'], parse_float=float)
                e_df = pd.Series(e_dict).reset_index().rename(columns={'index':'FID', 0:'err'})
                e_df['target'] = errors_data.iloc[sol]['target']
                e_df['var']  = single_best_sol['variable']
                e_df['is_normed']  = single_best_sol['normed']
                e_df['reg'] =  reg
                epz_pt = epz_pt.append(e_df, ignore_index=True)
            
    epz_pt.FID = epz_pt.FID.astype(int )
    return  epz_pt



def errors_per_FID_per_target(epz_pt, save_plot):
    fig, ax = plt.subplots(2,2, figsize=(20,20))
    for i in range(0,4):
        r,c  = compute_index_plot(4, i)
        
        if c == 1: reg = 'rfr'
        else: reg = 'svr'
        
        if r == 0: t_type =  'start'
        else: t_type  = 'final'
        
    #    for target in sorted(epz_pt.target.unique()):
    
        sol_df =  epz_pt[(epz_pt.reg == reg)\
                    &(epz_pt.target.str.contains(t_type))\
                    &(True)]
        
        for target in sorted(sol_df.target.unique()):
            sol_df_tp  = sol_df[sol_df.target==target]
            sol_df_tp = sol_df_tp.sort_values('FID')
            ax[r,c].plot(sol_df_tp.FID, sol_df_tp.err, label=target)
            ax[r,c].legend(ncol=2)
        ax[r,c].set_title('reg:%s, var:%s normed:%s'%(
                                            reg.upper(),
                                            sol_df_tp.iloc[0]['var'], 
                                            sol_df_tp.iloc[0]['is_normed']))
        ax[r,c].set_xticks(sorted(sol_df_tp.FID.unique()))
        ax[r,c].grid()
        ax[r,c].set_xlabel('Neighbour ID')
        ax[r,c].set_ylabel('Bookings Error [%]')
    if save_plot:
        plt.savefig(data_path+city+'/Regression/error_distirbution.pdf',\
                        bbox_incehs='tight', pad_inces=0)
        
        

def plot_mean_error_on_map(epz_pt, save_plot):
    filename =  'Vancouver_filtered_binned_2017-10-01T00-00-00_2017-10-31T23-59-59.csv'
    df = pd.read_csv(data_path+city+'/'+filename, nrows=None)
    limits = pd.read_csv(data_path+city+'/Vancouver_limits.csv')
    df = df[ (df.start_lat >= limits.min_lat.values[0])\
            &(df.end_lat >= limits.min_lat.values[0])\
            &(df.start_lon >= limits.min_lon.values[0])\
            &(df.end_lon >= limits.min_lon.values[0])\
            &(df.start_lat <= limits.max_lat.values[0])\
            &(df.end_lat <= limits.max_lat.values[0])\
            &(df.start_lon <= limits.max_lon.values[0])\
            &(df.end_lon <= limits.max_lon.values[0])
    ]
    gdf = df_coords2gdf(df, df.start_lat, df.start_lon)
    neighs = gpd.read_file(data_path\
                           +city\
                           +'/Opendata/Vancouver_macroArea/Vancouver_macroArea.shp')\
                           .to_crs({'init': 'epsg:4326'})[['MAPID', 'geometry']]
                           
                           
    mean_errors  = epz_pt.groupby(['FID']).mean()
    neighs['err'] = mean_errors['err']
#    neighs['coords'] = neighs['geometry'].apply(lambda x: x.representative_point().coords[:])
#    neighs['coords'] = [coords[0] for coords in neighs['coords']]
    neighs['coords'] = neighs.apply(lambda row: (row.geometry.centroid.x,
                                                 row.geometry.centroid.y), 
                                    axis=1)

    
    
    fig,ax = plt.subplots(1,1, figsize=(20,20))
    neighs.plot(column='err', ax=ax, legend=True)
    gdf.plot(ax=ax, markersize=0.1, color='red', alpha=0.8)
#    neighs.plot(edgecolor='black', color=None)
    
    for idx, row in neighs.iterrows():
        plt.annotate(s=str(idx),
                     xy=row['coords'],
                     horizontalalignment='center',
                     fontsize=8,
                     color='white'
                     )
    return neighs
    
        
        

     
        
        
        

if __name__=='__main__':
    
    city = 'Vancouver'
    data_path  = './../../data/'
    

                           
    res_rfr = data_path+city+'/Regression/output_rfr/rfr_regression_dist_fs.csv'
    res_svr = data_path+city+'/Regression/output_svr/svr_regression_dist_fs.csv'
    
    
    train = pd.read_csv('%s%s/Regression/dataset_train_emer.csv'%(data_path, city))
    valid = pd.read_csv('%s%s/Regression/dataset_test_emer.csv'%(data_path, city))
    complete_dataset = train.append(valid, ignore_index=True)
    
    stats = complete_dataset[complete_dataset.FID.isin([18,15,14,0,3])]
    pop_col = []
    for c in stats.columns:
        if 't_age' in c:
            pop_col.append(c)
    pop_df = stats[pop_col].T.sum().to_frame().rename(columns={0:'population'}).loc[[18,15,14,0,3]]
#    errors_df = create_errors_df(res_rfr, res_svr)
#    errors_df = errors_df[errors_df.err != '{}']
    
#    zzz = errors_df
#    best_sol = get_best_config(zzz)
    
    
#    epz_pt = compute_errors_per_FID_per_target(errors_df, best_sol)
#    errors_per_FID_per_target(epz_pt, save_plot=True)   
    
    
    

    
#        plt.annotate(s=row['NAME'], xy=row['coords'],
#                 horizontalalignment='center', 
#                 fontsize=5
#                 )
#        if idx % 5 == 0 and idx!=0: div = '\n'
#        else : div = '; '
#        title  +=str(idx) +'-'+ row['MAPID']+'-'+row['NAME'] + div
#    ax.set_title(title)



