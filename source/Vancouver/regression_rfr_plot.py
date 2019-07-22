#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:28:49 2019

@author: mc
"""

import matplotlib.pyplot as plt
import pandas as pd

city = 'Vancouver'
data_path  = './../../data/'

norm = False
df = pd.read_csv('%s%s/Regression/output_rfr/metrics_rfr_norm_%s.csv'%(data_path, city, norm))
# = ['RBF', 'Linear', 'Polynomial']

start_labels = []
final_labels = []
for c in df.target.unique():
    if 'c_start' in c:
        start_labels.append(c)
    if 'c_final' in c:
        final_labels.append(c)

start_labels = sorted(start_labels)
final_labels = sorted(final_labels)


for ne in df.n_estimators.unique():

    one_ne = df[df.n_estimators==ne]
    one_ne.loc[:,'err_r_valid'] = abs(one_ne.y_pred_valid - one_ne.y_valid )*100/one_ne.y_valid
    my_max = 0
    fig,ax=plt.subplots(2,1, figsize=(20,10)) 
    for ls, lf in  zip(start_labels, final_labels):
        df_s = one_ne[one_ne.target == ls]
        df_f = one_ne[one_ne.target == lf]
        
        #rescale
        df_s['y_valid_rescaled'] = df_s['y_valid']*df_s['std_target'] + df_s['mean_target']
        df_s['y_pred_valid_rescaled'] = df_s['y_pred_valid']*df_s['std_target'] + df_s['mean_target']
        df_s.loc[:,'err_r_valid_rescaled'] = abs(df_s.y_pred_valid_rescaled -\
                                                 df_s.y_valid_rescaled )*100/df_s.y_valid_rescaled

        
        df_f['y_valid_rescaled'] = df_f['y_valid']*df_f['std_target'] + df_f['mean_target']
        df_f['y_pred_valid_rescaled'] = df_f['y_pred_valid']*df_f['std_target'] + df_f['mean_target']
        df_f.loc[:,'err_r_valid_rescaled'] = abs(df_f.y_pred_valid_rescaled -\
                                                 df_f.y_valid_rescaled )*100/df_f.y_valid_rescaled
                
    
        metric = 'err_r_valid_rescaled' if norm == True else 'err_r_valid'
        my_max =  max(my_max,
                      max(df_s[metric].max(),  df_f[metric].max())
                      )
        
#        print (my_max)
        

        
        ax[0].set_title('N_estimators = %d\nInit dataset were normed: %s\nmean_s:%f median_s:%f std_s:%f\nmean_f:%f median_f:%f std_f:%f'%(ne, norm, 
          df_s[metric].mean(),  df_s[metric].median(), df_s[metric].std(),
          df_f[metric].mean(),  df_f[metric].median(), df_f[metric].std(),))
        ax[0].plot(df_s.FID_valid, df_s[metric], label=ls)
        ax[1].plot(df_f.FID_valid, df_f[metric], label=lf)


    max_ytick  = (int(my_max/100) +1) *100
    ax[0].set_xticks(range(0,21))
    ax[0].set_yticks(range(0,max_ytick,100))
    ax[0].set_ylabel(metric)
    ax[0].grid()
    ax[0].legend()
#    ax[0].set_xlabel('FID used like valid')
    
    ax[1].set_xticks(range(0,21))
    ax[1].set_yticks(range(0,max_ytick,100))
    ax[1].set_ylabel(metric)
    ax[1].set_xlabel('FID used like valid')
    ax[1].grid()
    ax[1].legend()
    
    plt.savefig('%s%s/Regression/output_rfr/%d_%s.pdf'%(data_path, city, ne, norm))