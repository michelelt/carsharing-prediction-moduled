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
        compute_mean_err_perc\



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
    
    if normed:
        valid = 'rb_y_valid'
        pred  = 'rb_y_pred'
    else:
        valid = 'y_valid'
        pred = 'y_pred_valid'
    err_perc = compute_mean_err_perc(average_mean_error[pred],
                                     average_mean_error[valid])*100
        

    return err_perc



def configs_learning_curve_rfr(errors_rfr, reg_type, normed, variable, targets):
    
    if reg_type == 'rfr':
        n_estimators = variable
        error_df = get_best_rfr_df(errors_rfr, normed, n_estimators, targets)
    else:
        kernel = variable
        error_df = get_best_rfr_df(errors_rfr, normed, kernel, targets)
    
    if error_df is None:
        print('Some Errors')
        return
        
    len(error_df)
    ame = []
    
#    fig,ax = plt.subplots(figsize=(20,10))
    data = {}
    for target in compute_target_labels()[targets]:
        for nof in range(1,84):
            ame.append(compute_average_mean_error(nof, normed, target, error_df))
            
        data[target] = {
                'x':  range(1,84),
                'y': ame,
                'target':  target
                }
            
            
        
#        ax.plot(range(1,84), ame, label=target)
#        ax.legend(ncol=7, loc='upper center')
#        ax.set_xlabel('Numner of selected features')
#        ax.set_ylabel('Mean average error')
#        ax.grid()
#        ax.set_title('Reg. type: %s; Normalized: %s; Varbiale: %s'%(
#                reg_type.upper(), str(normed), variable))
        
        ame=[]

    
#    if save_plot:
#        plt.savefig(data_path+city+'/Regression/output_%s/LC_%s_%s.pdf'%\
#                (reg_type, str(variable), str(normed)),
#                bbox_incehs='tight', pad_inces=0)
        
    return [reg_type, normed, variable, compute_target_labels()[targets], data ]



def plot_learning_curves_4(save_plot):
    configs = []
    configs.append(configs_learning_curve_rfr(errors_rfr, 'rfr', True, 30, 'starts'))
    configs.append(configs_learning_curve_rfr(errors_rfr, 'rfr', True, 80, 'finals'))
    configs.append(configs_learning_curve_rfr(errors_svr, 'svr', True, 'poly', 'starts'))
    configs.append(configs_learning_curve_rfr(errors_svr, 'svr', False, 'linear', 'finals'))
    #
    #
    #
    fig, ax = plt.subplots(2,2, figsize=(20,20))
    i_config = 0
    for i in range(0,2):
        for j in range(0,2):
    ##        
            reg_type = configs[i_config][0]
            normed = configs[i_config][1]
            variable = configs[i_config][2]
            targets  = configs[i_config][3]
    #        
            for target in targets:
                x = configs[i_config][4][target]['x']
                y = configs[i_config][4][target]['y']
    
                ax[i,j].plot(x, y, label=target)
                ax[i,j].set_xlabel('Numbner of selected features')
                ax[i,j].set_ylabel('Mean average error')
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
        plt.savefig(data_path+city+'/Regression/LC_.pdf',\
                        bbox_incehs='tight', pad_inces=0)
    

src_svr = '/home/det_user/cocca/BETA_copy/data/Vancouver/Regression/output_svr/'
filename_svr = 'svr_regression_fs.csv'

src_rfr = '/home/det_user/cocca/BETA_copy/data/Vancouver/Regression/output_rfr/'
filename_rfr = 'rfr_regression_fs.csv'

src_svr += filename_svr
src_rfr += filename_rfr


city = 'Vancouver'
data_path  = './../../data/'
dst_svr = data_path+city+'/Regression/output_svr'
dst_rfr = data_path+city+'/Regression/output_rfr'



#ssh_client = ssh_connection()
#download_file(ssh_client, src, dst_svr)

#errors_rfr = pd.read_csv(dst_rfr+'/'+filename_rfr)
#errors_svr = pd.read_csv(dst_svr+'/'+filename_svr)
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

plot_learning_curves_4(True)

          
        








