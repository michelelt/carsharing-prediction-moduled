#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 00:47:02 2019

@author: mc
"""
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')
from GlobalsFunctions import crs_

import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from GlobalsFunctions import *
import io, pkgutil    
import geoplot 
from matplotlib import colors   


data_path = '../../data/Vancouver/Regression/'
ind_var = 'time'
file = 'Vancouver_perf_%s.csv'%ind_var
df = pd.read_csv(data_path+file)





starts = ['c_start_%d'%tb for tb in range(0,7)]
finals = ['c_final_%d'%tb for tb in range(0,7)]
starts_df = df[df.Label.isin(starts)]
finals_df = df[df.Label.isin(finals)]
metrics =  ['F-value','MSE', 'RSS', 'RSE', 'R2', 'autocorr_err', 'feat sig', 'err_abs_rescaled']

def get_csvs(df, nof, label):
    
    line = df[(df.Label  == label) &  (df.NOF==nof)]
    
    y_pred_str = str.encode(
        'FID,%s\n%s'%\
        (line.iloc[0]['Label'], line.iloc[0]['y_pred'])
        )
    y_pred = pd.read_csv(io.BytesIO(y_pred_str),  encoding='utf-8' )
    
    
    y_pred_str = str.encode(
        'FID,%s\n%s'%\
        (line.iloc[0]['Label'], line.iloc[0]['y_test'])
        )
    y_test = pd.read_csv(io.BytesIO(y_pred_str),  encoding='utf-8' )
    
        
    pvalues_str = str.encode(
        (line.iloc[0]['P-values'])
        )
    pvalues = pd.read_csv(io.BytesIO(pvalues_str), encoding='utf-8', sep=';')\
                .rename(columns={"value":'P-values'})
    
    
    tvalues_str = str.encode(
        (line.iloc[0]['T-values'])
        )
    tvalues = pd.read_csv(io.BytesIO(tvalues_str), encoding='utf-8', sep=';')\
                .rename(columns={"value":'T-values'})
                
    pvalues.loc[:, 'T-values'] = tvalues['T-values']
    return y_pred, y_test, pvalues, line.iloc[0]['y_test_mean'], line.iloc[0]['y_test_std']
#
#
#    
#    
#
for metric in  ['F-value']:
#for metric in metrics:
##    metric='RSE'
    fig, ax = plt.subplots(2,1, figsize=(20,10))
    x = df.NOF.unique()
    for i  in  range(0,7):
        
        
        to_plot = starts_df[starts_df.Label==starts[i]]
        ax[0].plot(x, to_plot[metric], label=starts[i])
        ax[0].legend(ncol=7)
        if metric == 'err_abs_rescaled':
            ax[0].set_title(metric+'\nsum(abs(y_test*std - y_pred*std))', fontsize=20)
        else:
            ax[0].set_title(metric)
        
        to_plot = finals_df[finals_df.Label==finals[i]]
        ax[1].plot(x, to_plot[metric], label=finals[i])
        ax[1].legend(ncol=7)
        ax[1].set_xlabel('Number of features in the model')
        
            
#        
    plt.savefig(metric+'.pdf', bbox_inches='tight')
#        

## =============================================================================
## '''
## PLOT the error heatmap for each tile
## '''
## =============================================================================
#test = pd.read_csv('./../../data/Vancouver/Regression/test_emer.csv').fillna(0)
#test['geometry'] = test.apply(lambda x: str2polygon(x.geometry), axis =1)
#test = gpd.GeoDataFrame(test.drop('geometry', axis=1),
#                        geometry = test.geometry,
#                        crs=crs_)
#test = test.set_index('FID')
#fig, ax = plt.subplots(nrows=4,ncols=4, figsize=(35,10))
#fig.tight_layout()
#vmin = 807.7299605735902
#vmax = -27.43656631218449
#images = []
#cmap ='seismic'
#for i in range(0,7):
#    analyzed_label = 'c_start_%d'%i
#    y_pred, y_test, ptest, mean, std = get_csvs(df, 100, analyzed_label)
#    city='Vancouver'
#    
#    y_pred[analyzed_label] = y_pred[analyzed_label].mul(std).add(mean)
#    y_test[analyzed_label] = y_test[analyzed_label].mul(std).add(mean)
#    err = y_pred.set_index('FID') - y_test.set_index('FID')
#    
#    test
#    test['err_rel_%s'%analyzed_label] = err
#    test[analyzed_label] = y_test.set_index('FID')
#    if i > 3 :row =1
#    else: row=0
#    test.plot(
#            ax = ax[row,i%4],
#            column='err_rel_%s'%analyzed_label,
#            cmap=cmap,
#            legend=True
#            )
#    
#
#    ax[row, i%4].set_title('err_abs_%s'%analyzed_label)
#    ax[row, i%4].set_xticks([])
#    ax[row, i%4].set_yticks([])
#    ax[row, i%4].axis('off')
#    
#    vmin = min(vmin, y_pred[analyzed_label].min(), y_test[analyzed_label].min() )
#    vmax = max(vmax, y_pred[analyzed_label].max(), y_test[analyzed_label].max() )
#    
#    
#    analyzed_label = 'c_final_%d'%i
#    y_pred, y_test, ptest, mean, std = get_csvs(df, 100, analyzed_label)
#    
#    y_pred[analyzed_label] = y_pred[analyzed_label].mul(std).add(mean)
#    y_test[analyzed_label] = y_test[analyzed_label].mul(std).add(mean)
#    err = y_pred.set_index('FID') - y_test.set_index('FID')*1
#    
#    test['err_rel_%s'%analyzed_label] = err
#    test[analyzed_label] = y_test.set_index('FID')
#    if i > 3 :row =3
#    else: row=2
#    test.plot(
#            ax = ax[row,i%4],
#            column='err_rel_%s'%analyzed_label,
#            cmap=cmap,
#            legend=True
#            )
#    
#
#    ax[row, i%4].set_title('err_abs_%s'%analyzed_label)
#    ax[row, i%4].set_xticks([])
#    ax[row, i%4].set_yticks([])
#    ax[row, i%4].axis('off')
#
#    
#    vmin = min(vmin, y_pred[analyzed_label].min(), y_test[analyzed_label].min() )
#    vmax = max(vmax, y_pred[analyzed_label].max(), y_test[analyzed_label].max() )
    
#    images.append(ax[0, i].imshow(test['err_abs_%s'%analyzed_label].values, cmap=cmap))
#    images.append(ax[1, i].imshow(test['err_abs_%s'%analyzed_label].values, cmap=cmap))
#i=7
#ax[1, i%4].axis('off')
#ax[3, i%4].axis('off')
#fig.subplots_adjust(hspace=0.0)












































