#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:15:32 2019

@author: mc
"""

import pandas as pd

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../../') 

import matplotlib.pyplot as plt


from GlobalsFunctions import ssh_connection,\
        download_file,\
        compute_target_labels,\
        compute_mean_err_perc,\
        compute_rmse,\
        create_errors_df,\
        get_best_config
        
import json

type_of_error={
        'err_mean_perc':0,
        'rmse':1
        }

'''
learning curve for starts
'''

def get_best_rfr_df(errors_rfr, normed, variable, target):
    print(variable)
    print(type(variable))
    if type(variable) is str:
        var_to_search = 'kernel'
    elif type(variable) is int :
        var_to_search = 'n_estimators'
    else:
        return None
        
    targets = compute_target_labels()[target]
    best_rfr = errors_rfr[True\
            & (errors_rfr.is_normed == normed)\
            & (errors_rfr[var_to_search] == variable)\
            & (errors_rfr.target.isin(targets))
            ]
    
    return best_rfr


def compute_average_mean_error(nof, normed, target, df):
    average_mean_error =  df[True\
            &(df.nof == nof )\
            &(df.target == target)
            ]        
    err_perc = average_mean_error['err_mean_perc'].values
    rmse = average_mean_error['rmse'].values
    return [err_perc,rmse]



def configs_learning_curve(errors_df, reg_type, normed, variable, targets, error_name):
    errors_df = errors_df[True\
                &(errors_df.reg_type==reg_type)\
                &(errors_df.is_normed==normed)\
            ]
    
    if reg_type == 'rfr':
        n_estimators = variable
        error_df = errors_df[errors_df.n_estim==n_estimators]
    else:
        kernel = variable
        error_df = errors_df[errors_df.kernel==kernel]
    
    if error_df is None:
        print('Some Errors')
        return
        
    ame = []
    
#    fig,ax = plt.subplots(figsize=(20,10))
    data = {}
    for target in compute_target_labels()[targets]:
        for nof in range(1,len(errors_df.nof.unique())):
            ame.append(compute_average_mean_error(nof, normed, target, error_df)[type_of_error[error_name]])
            
        data[target] = {
                'x':  range(1,len(errors_df.nof.unique())),
                'y': ame,
                'target':  target
                }
        
        ame=[]

    
        
    return [reg_type, normed, variable, compute_target_labels()[targets], data ]



def plot_learning_curves_4(save_plot, errors_df, best_sols, error_name):
    configs = []
    configs.append(configs_learning_curve(errors_df, 
                                              'rfr', 
                                              best_sol['rfr']['start']['normed'],
                                              best_sol['rfr']['start']['variable'], 
                                              'starts',
                                              error_name)
        )
    configs.append(configs_learning_curve(errors_df, 'rfr', 
                                              best_sol['rfr']['final']['normed'], 
                                              best_sol['rfr']['final']['variable'], 
                                              'finals',
                                              error_name)
        )
    configs.append(configs_learning_curve(errors_df, 'svr', 
                                              best_sol['svr']['start']['normed'], 
                                              best_sol['svr']['start']['variable'], 
                                              'starts',
                                              error_name)
        )
    
    configs.append(configs_learning_curve(errors_df, 'svr', 
                                              best_sol['svr']['final']['normed'], 
                                              best_sol['svr']['final']['variable'],
                                              'finals',
                                              error_name)
        )

    fig, ax = plt.subplots(2,2, figsize=(20,20))
    
    i_config = 0
    for i in range(0,2):
        for j in range(0,2):
    ##        
            reg_type = configs[i_config][0]
            normed = configs[i_config][1]
            variable = configs[i_config][2]
            targets  = configs[i_config][3]
            for target in targets:
                x = configs[i_config][4][target]['x']
                y = configs[i_config][4][target]['y']
                
                ax[i,j].plot(x, y, label=target)
                ax[i,j].set_xlabel('Numbner of selected features')
                ax[i,j].set_ylabel(error_name)
                ax[i,j].grid()
                ax[i,j].set_title('Reg. type: %s; Normalized: %s; Varbiale: %s'%(
                        reg_type.upper(), str(normed), variable)) 
            ax[i,j].legend(ncol=4, loc='upper center')
            
            i_config+=1
    
    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    
    ax[0,0].set_xlabel('')
    ax[0,1].set_xlabel('') 
    
    if save_plot:
        print('Plot saved')
        plt.savefig(data_path+city+'/Regression/LC_.pdf',\
                        bbox_incehs='tight', pad_inces=0)
        
    return 1
    

src_svr = '/home/det_user/cocca/BETA_copy/data/Vancouver/Regression/output_svr/'
filename_svr = 'svr_regression_dist_fs.csv'

src_rfr = '/home/det_user/cocca/BETA_copy/data/Vancouver/Regression/output_rfr/'
filename_rfr = 'rfr_regression_dist_fs.csv'

src_svr += filename_svr
src_rfr += filename_rfr


city = 'Vancouver'
data_path  = './../../data/'
dst_svr = data_path+city+'/Regression/output_svr'
dst_rfr = data_path+city+'/Regression/output_rfr'



#ssh_client = ssh_connection()
#download_file(ssh_client, src_svr, dst_svr)
#download_file(ssh_client, src_rfr, dst_rfr)


errors_rfr = pd.read_csv(dst_rfr+'/'+filename_rfr)
errors_svr = pd.read_csv(dst_svr+'/'+filename_svr)
#print('is normed',    len(errors_rfr.is_normed.unique()  )  ) 
#print('targets',      len(errors_rfr.target.unique() )  )
#print('n_estimators', len(errors_rfr.n_estimators.unique() ) )
#print('FID_valid',    len(errors_rfr.FID_valid.unique() ) )
#print('nof',          len(errors_rfr.nof.unique() ) ) 
#
#
#
#a =  len(errors_rfr.is_normed.unique())
#a *= len(errors_rfr.target.unique() )  
#a *= len(errors_rfr.n_estimators.unique() )
#a *= len(errors_rfr.FID_valid.unique() )
#a *= len(errors_rfr.nof.unique() ) 


errors_df = create_errors_df(dst_rfr + '/' + filename_rfr, 
                             dst_svr + '/' + filename_svr)
best_sol = get_best_config(errors_df)
#
zzz = errors_df[errors_df.err != '{}']
line_nan=zzz[zzz.err != '{}']

aaa = plot_learning_curves_4(True, zzz, best_sol, 'err_mean_perc')   



#def compute_feature_ranks(res_rfr):
#
#    rfr_df =  pd.read_csv(res_rfr)
#    ranks_list = []
#    for rank_str in rfr_df['rank']:
#        ranks_list.append(json.loads(rank_str)['score'])
#    ranks_df = pd.DataFrame(ranks_list)
#    
#    mean_ranks   = ranks_df.mean().sort_values(ascending=False)
#    median_ranks = ranks_df.median().sort_values(ascending=False)
#    sum_ranks    = ranks_df.sum().sort_values(ascending=False)
#    
#    dfs = [mean_ranks,  median_ranks, sum_ranks]
##    labels = ['Mean', 'Median', 'Best']
##    colors = ['blue', 'green', 'red']
#        
#    return {'mean': mean_ranks, 'median': median_ranks,  'sum': sum_ranks}
#most_ranked = compute_feature_ranks( dst_rfr + '/rfr_regression_dist.csv')['mean']
#
#for nof in range(1, len(most_ranked)+1):
#    z = most_ranked.iloc[0:nof+1]
                                                                   	           
