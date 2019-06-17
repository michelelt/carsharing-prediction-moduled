#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:54:13 2019

@author: mc
"""

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt
#
import geopandas as gpd
import pandas as pd


def compute_cdf(y_set):
    sorted_data = np.sort(y_set)
    yvals = np.arange(len(sorted_data))/(len(y_set)-1)
    
    return [sorted_data, yvals]

data_path = './../data/'
city = 'Vancouver'


train = pd.read_csv(data_path+city+'/Regression/train.csv')
test  = pd.read_csv(data_path+city+'/Regression/test.csv')

fid_train_not_in_fid_test = []
for fid in train.FID:
    if fid not in list(test.FID): 
        fid_train_not_in_fid_test.append(fid)
        
fid_test_not_in_fid_train = []
for fid in train.FID:
    if fid not in list(train.FID): 
        fid_test_not_in_fid_train.append(fid)
        
        
'''
makes both dataset equal number of tiles
'''

train = train.set_index('FID').drop(fid_train_not_in_fid_test).reset_index()
test  = test.set_index('FID').drop(fid_test_not_in_fid_train).reset_index()
test_mean =train.mean()
test_std = train.std()
        
       


columns_to_delete=[
'MAPID', 'FID', 'lat', 'lon', 'geometry_nwf', 'geometry_neigh'
        ]
train_norm = train.drop(columns_to_delete, axis=1)
new_columns = train_norm.columns
train_norm = (train_norm - train_norm.mean())/train_norm.std()

corr_df = train_norm.astype(float)
corr = train_norm.corr()

row_to_del =[]
col_to_keep=[]
for c in train.columns:
#    if 'start' in c or 'final'  in c \ #comment to consider in the corr mat
    if ('sum' in c)   or ('count' in  c)\
    or ('Gi_' in c) :
        row_to_del.append(c)

    if 'start' in c or 'final'  in c: col_to_keep.append(c)
#    
corr = corr.loc[corr.columns.difference(row_to_del)][col_to_keep]


test_norm = test.drop(columns_to_delete, axis=1)
new_columns = test_norm.columns
test_norm = (test_norm - test_norm.mean())/test_norm.std()

#fig,ax  = plt.subplots(7,2)
#for tb in range(0,7):
#    label = 'c_start_%d'%tb
#    x,y = compute_cdf(corr[label])
#    ax[tb,0].plot(x,y, label = label)
#    ax[tb,0].grid()
#    ax[tb,0].legend()
#    
#    label = 'c_final_%d'%tb
#    x,y = compute_cdf(corr[label])
#    ax[tb,1].plot(x,y, label = label)
#    ax[tb,1].grid()
#    ax[tb,1].legend()
#
#
#

#from sklearn.linear_model import LinearRegression
#
#for tb in range(0,7):
#    for nof in range(1,21,5):
#        prediction_label = 'c_start_%d'%tb
#        y_train = train_norm[prediction_label].values
#        
#        df_temp = corr[prediction_label].sort_values(ascending=False).iloc[1:nof+1]
#        X_train = train_norm[df_temp.index].astype(float)
#        
#        reg = LinearRegression().fit(X_train,y_train)
#        print('tb:%d, #F:%d, R^2=%f' % (tb, nof, reg.score(X_train,y_train)))
#              
#        '''
#        prediction experiments
#        '''
#        X_test = test_norm[df_temp.index].astype(float).values
#        y_test = test_norm[prediction_label]
#        y_pred = np.matmul(X_test,reg.coef_)
#        
#        y_pred =( y_pred * test_std[prediction_label]) + test_mean[prediction_label]
#        y_test =( y_test * test_std[prediction_label]) + test_mean[prediction_label]
#        err =  np.abs((y_test-y_pred)) / y_test
#
#        
#        '''
#        Plot
#        '''
#        fig, ax = plt.subplots(1,3,figsize=(21,7))
#        ax[0].set_title('Prediction %s nof: %d'%(prediction_label, nof) )
#        ax[0].plot(y_pred, label='pred', linestyle='--')
#        ax[0].plot(y_test.values, label='test', linestyle=':')
#        ax[0].legend()
#
#        
#        ax[1].set_title("Error perc. %s nof: %d"%(prediction_label, nof))
#        print('err perc, media mediana: %.2f %.2f'%( err.mean(), err.median()))
#        ax[1].plot(err)
#        ax[1].set_ylim([0,100])
#        
#        
##            
#    print()
##    
##    
#    

#    
#    
#    
#    
#    
#    
