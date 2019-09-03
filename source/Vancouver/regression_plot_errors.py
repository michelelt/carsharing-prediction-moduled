#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:03:44 2019

@author: mc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../')  

from GlobalsFunctions import create_errors_df,\
    get_best_config,\
    ssh_connection,\
    download_file,\
    compute_feature_rank

import json




def plot_top_n_features(n, ranks_df, labels, save_plot=False):
        
    if n is None:  n=len(ranks_df)
    mean_ranks   = ranks_df.mean().sort_values(ascending=False).iloc[0:n]
    median_ranks = ranks_df.median().sort_values(ascending=False).iloc[0:n]
    sum_ranks    = ranks_df.sum().sort_values(ascending=False).iloc[0:n]
    
    dfs = [mean_ranks,  median_ranks, sum_ranks]
#    labels = ['Mean', 'Median', 'Best']
#    colors = ['blue', 'green', 'red']

    
    fig, ax = plt.subplots(len(labels), 1, figsize=(10, 30))
    if len(labels) == 1: axs=[ax]
    for i in range(len(labels)):
        print(i)
        ind = np.arange(len(mean_ranks))  # the x locations for the groups
        width = 0.35  # the width of the bars
        axs[i].barh(ind,  dfs[i], width, label=labels[i])
        
        ticklabels = dfs[i].reset_index()['index'].tolist()
        for t, element in enumerate(ticklabels):
            ticklabels[t] =  element.replace('$', '\$')
            
        axs[i].set_yticks(ind)
        if n >30:
            axs[i].set_yticklabels(ticklabels,  rotation=0, ha='right', fontsize=5)
        else:
            axs[i].set_yticklabels(ticklabels,  rotation=0, ha='right')
        axs[i].invert_yaxis()
        axs[i].legend()
        
    if save_plot:
        print(data_path+city+'/Regression/feature_ranks.pdf')

        plt.savefig(data_path+city+'/Regression/feature_ranks.pdf',
                    bbox_inches = 'tight')
        
    return {'mean': mean_ranks, 'median': median_ranks,  'sum': sum_ranks}


        
    
def plot_errors_per_regression(errors_df,
                               is_normed,
                               reg_type,
                               want_medians,
                               param1,
                               param2,
                               save_plot=False):

    
    
    starts = errors_df[errors_df\
                       .target\
                       .str.contains('start')]['target']\
                       .unique().tolist()
                       
    finals = errors_df[errors_df\
                       .target\
                       .str\
                       .contains('final')]['target']\
                       .unique().tolist()
                       
    if want_medians:
        configs = [
                [0,0,  starts, param1],
                [0,1,  starts, param2],
                [1,0,  finals, param1],
                [1,1,  finals, param2],
                ]
        nrows, ncols = 2,2
    else:
        configs = [
                [0,0,  starts, param1],
                [1,0,  finals, param1],
                ]
        nrows, ncols = 2,1
    

    
    xax_metric='kernel' if reg_type == 'svr' else 'n_estim'
    xlabel = 'Kernel' if reg_type == 'svr' else 'Number of estimators'
    
    fig, ax = plt.subplots(nrows,ncols, figsize=(40,40))
    
    fig.suptitle('Regression type: %s\ndataset is normalzied: %s'% (reg_type.upper(), str(is_normed))   )
    
    for config in configs:
        row = config[0]
        col = config[1]
        targets =  config[2]
        metric = config[3]
        
        for target in targets:
            to_plot =  errors_df[ (errors_df.target == target)\
                                 &(errors_df.is_normed==is_normed)
                                 &(errors_df.reg_type== reg_type) ]
#            return to_plot
            
            if want_medians:
                print(to_plot[xax_metric],  to_plot[metric])
                ax[row, col].plot(sorted(to_plot[xax_metric]), to_plot[metric], label=target)
                ax[row, col].grid()
                ax[row, col].set_xticks(sorted(to_plot[xax_metric]))
                ax[row, col].set_xlabel(xlabel)
                ax[row, col].set_ylabel(metric)
            else:
                ax[row].plot(sorted(to_plot[xax_metric]), to_plot[metric], label=target)
                ax[row].grid()
                ax[row].set_xticks(sorted(to_plot[xax_metric]))
                ax[row].set_xlabel(xlabel)
                ax[row].set_ylabel(metric)
                
        
        
        if want_medians: ax[row,col].legend(ncol=4)
        else: ax[row].legend(ncol=4)
        
    if save_plot:
        plt.savefig(data_path+city+'/Regression/output_%s/MAE_%s_%s.pdf'%(reg_type, reg_type, str(is_normed)),
                    bbox_inches = 'tight')
 

def plot_avg_err_per_nestim(errors_df,
                               is_normed,
                               reg_type, 
                               param1,
                               param2):

    starts = errors_df[errors_df\
                       .target\
                       .str.contains('start')]['target']\
                       .unique().tolist()
                       
    finals = errors_df[errors_df\
                       .target\
                       .str\
                       .contains('final')]['target']\
                       .unique().tolist()
    configs = [
            [0,0,  starts, param1],
            [0,1,  starts, param2],
            [1,0,  finals, param1],
            [1,1,  finals, param2],
            ]
    
    
    xax_metric='kernel' if reg_type == 'svr' else 'n_estim'
    xlabel = 'Kernel' if reg_type == 'svr' else 'Number of estimators'
    
    fig, ax = plt.subplots(2,2, figsize=(40,40))
    
    fig.suptitle('Regression type: %s\ndataset is normalzied: %s'% (reg_type.upper(), 
                                                                    str(is_normed))   )
    
    for config in configs:
        row = config[0]
        col = config[1]
        targets =  config[2]
        metric = config[3]
        label = 'Mean Error of %s' % metric
        
#        for target in targets:
        to_plot =  errors_df[(errors_df.target.isin(targets))\
                             &(errors_df.is_normed==is_normed)
                             &(errors_df.reg_type== reg_type) ]
        
        to_plot_grouped = to_plot.groupby(xax_metric).mean().reset_index()
        
        if 'mean' in metric:
            print(reg_type, targets[0][2:-2],
                  metric, is_normed,  to_plot_grouped[metric].min(),
                  to_plot_grouped.loc[to_plot_grouped[metric].idxmin(),xax_metric])
        
        ax[row, col].plot(sorted(to_plot_grouped[xax_metric]), to_plot_grouped[metric], label=label)
        ax[row, col].grid()
        ax[row, col].set_xticks(sorted(to_plot_grouped[xax_metric]))
        ax[row, col].set_xlabel(xlabel)
        ax[row, col].set_ylabel(metric)
        
        
    #    break
        ax[row,col].legend(ncol=4)
        

    




''''''

city = 'Vancouver'
data_path  = './../../data/'

filename_svr = 'svr_regression_dist.csv'
filename_rfr = 'rfr_regression_dist.csv'

dst_svr = data_path+city+'/Regression/output_svr'
dst_rfr = data_path+city+'/Regression/output_rfr'




res_rfr = dst_rfr+'/'+filename_rfr
res_svr = dst_svr+'/'+filename_svr
rfr = pd.read_csv(res_rfr)
svr = pd.read_csv(res_svr)


errors_df = create_errors_df(res_rfr, res_svr)
best_sol = get_best_config(errors_df)



SP=True
want_median=True
ranks_df = compute_feature_rank(res_rfr,True,data_path+city+'/Regression/feature_ranks.csv')
ranks = plot_top_n_features(None, ranks_df, ['Mean'], save_plot=SP)

#param1='err_mean_perc'
#param2='rmse'
#plot_errors_per_regression(errors_df, True, 'svr', want_median, 
#                           param1, param2, save_plot=SP)
#plot_errors_per_regression(errors_df, False, 'svr', want_median,
#                            param1, param2, save_plot=SP)
#
#
#plot_errors_per_regression(errors_df, True, 'rfr', want_median, 
#                            param1, param2, save_plot=SP)
#plot_errors_per_regression(errors_df, False, 'rfr', want_median, 
#                            param1, param2, save_plot=SP)

#
#
#
#plot_avg_err_per_nestim(errors_df, True, 'svr', param1, param2)
#plot_avg_err_per_nestim(errors_df, False, 'svr', param1, param2)
#plot_avg_err_per_nestim(errors_df, True, 'rfr', param1, param2)
#plot_avg_err_per_nestim(errors_df, False, 'rfr', param1, param2)



    
    
    