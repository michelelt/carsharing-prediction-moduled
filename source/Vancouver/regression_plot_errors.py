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

import json

city = 'Vancouver'
data_path  = './../../data/'
res_rfr = data_path+city+'/Regression/output_rfr/rfr_regression.csv'
res_svr = data_path+city+'/Regression/output_svr/svr_regression.csv'


def plot_top_n_features(n, res_rfr):
    
    rfr_df =  pd.read_csv(res_rfr)
    ranks_list = []
    for rank_str in rfr_df['rank']:
        ranks_list.append(json.loads(rank_str)['score'])
    ranks_df = pd.DataFrame(ranks_list)
    
    if n is None:  n=len(ranks_df)
    mean_ranks   = ranks_df.mean().sort_values(ascending=False).iloc[0:n]
    median_ranks = ranks_df.median().sort_values(ascending=False).iloc[0:n]
    sum_ranks    = ranks_df.sum().sort_values(ascending=False).iloc[0:n]
    
    dfs = [mean_ranks,  median_ranks, sum_ranks]
    labels = ['Mean', 'Median', 'Best']
#    colors = ['blue', 'green', 'red']
    
    

    
    fig, ax = plt.subplots(3,1, figsize=(10, 30))
    for i, rank in enumerate(dfs):
        print(i)
        ind = np.arange(len(mean_ranks))  # the x locations for the groups
        width = 0.35  # the width of the bars
        ax[i].barh(ind, rank, width, label=labels[i])
        
        ticklabels = rank.reset_index()['index'].tolist()
        for t, element in enumerate(ticklabels):
            ticklabels[t] =  element.replace('$', '\$')
            
        ax[i].set_yticks(ind)
        ax[i].set_yticklabels(ticklabels,  rotation=0, ha='right', fontsize=5)
        ax[i].invert_yaxis()
        ax[i].legend()



def create_errors_df(res_rfr, res_svr):

    rfr_df =  pd.read_csv(res_rfr).drop('rank', axis=1)
    svr_df = pd.read_csv(res_svr)
    
    errors_list = []
    for is_normed in rfr_df.is_normed.unique():
        for target in rfr_df.target.unique():
            for n_estim in rfr_df.n_estimators.unique():
    #            normed = True
    #            target = 'c_start_0'
    #            n_estim = 40
                
                a = rfr_df[rfr_df.is_normed == is_normed]
                a = a[a.target==target]
                a = a[a.n_estimators == n_estim]
                a = a.set_index('FID_valid')
                if is_normed:
                    y_pred_label = 'rb_y_pred'
                    y_valid_label = 'rb_y_valid'
                else:
                    y_pred_label = 'y_pred_valid'
                    y_valid_label = 'y_valid'            
                    
                    
                err_perc = abs((a[y_pred_label] - a[y_valid_label])*100).div(a[y_valid_label])
    
                errors_list.append(
                        { 'is_normed':is_normed,
                          'target':target,
                          'n_estim':n_estim,
                          'err_mean': err_perc.mean(),
                          'err_median':err_perc.median(),
                          'err': err_perc.to_json(),
                          'reg_type': 'rfr'
                        })
                
    
    for is_normed in svr_df.is_normed.unique():
        for kernel in svr_df.kernel.unique():
            for target in svr_df.target.unique():
                a = svr_df[svr_df.is_normed == is_normed]
                a = a[a.kernel == kernel] 
                a = a[a.target == target]
                
                if is_normed:
                    y_pred_label = 'rb_y_pred'
                    y_valid_label = 'rb_y_valid'
                else:
                    y_pred_label = 'y_pred_valid'
                    y_valid_label = 'y_valid'            
                    
                    
                err_perc = abs((a[y_pred_label] - a[y_valid_label])*100).div(a[y_valid_label])
                
                errors_list.append(
                        { 'is_normed':is_normed,
                          'target':target,
                          'kernel':kernel,
                          'err_mean': err_perc.mean(),
                          'err_median':err_perc.median(),
                          'err': err_perc.to_json(),
                          'reg_type': 'svr'
                        })
        
    errors_df = pd.DataFrame(errors_list)
    return errors_df
        
    
def plot_errors_per_regression(errors_df,
                               is_normed,
                               reg_type,
                               ):

    
    
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
            [0,0,  starts, 'err_mean'],
            [0,1,  starts, 'err_median'],
            [1,0,  finals, 'err_mean'],
            [1,1,  finals, 'err_median'],
            ]
    
    xax_metric='kernel' if reg_type == 'svr' else 'n_estim'
    xlabel = 'Kernel' if reg_type == 'svr' else 'Number of estimators'
    
    fig, ax = plt.subplots(2,2, figsize=(40,40))
    
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
            
            print(to_plot.shape)
            
            
            ax[row, col].plot(to_plot[xax_metric], to_plot[metric], label=target)
            ax[row, col].grid()
            ax[row, col].set_xticks(to_plot[xax_metric])
            ax[row, col].set_xlabel(xlabel)
            ax[row, col].set_ylabel(metric)
        
        
    #    break
        ax[row,col].legend(ncol=4)
 

#errors_df = create_errors_df(res_rfr, res_svr)
#
#plot_top_n_features(15, res_rfr)
#plot_errors_per_regression(errors_df, True, 'svr')
#plot_errors_per_regression(errors_df, False, 'svr')
#plot_errors_per_regression(errors_df, True, 'rfr')
#plot_errors_per_regression(errors_df, False, 'rfr')
#


    
    
    


rfr_df =  pd.read_csv(res_rfr)
ranks_list = []
for rank_str in rfr_df['rank']:
    ranks_list.append(json.loads(rank_str)['score'])
ranks_df = pd.DataFrame(ranks_list)

ranks_df.to_csv(data_path+city+'/Regression/feature_ranks.csv', index=False)