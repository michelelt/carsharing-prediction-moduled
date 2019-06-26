#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:39:23 2019

@author: mc
"""

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


train = pd.read_csv(data_path+city+'/Regression/train_emer.csv').fillna(0)
#if 'geometry' in train.columns: train = train.drop('geometry', axis=1)
test  = pd.read_csv(data_path+city+'/Regression/test_emer.csv')


# =============================================================================
# Detect which FID1 tiles are not in FID2
# =============================================================================
fid_train_not_in_fid_test = []
for fid in train.FID:
    if fid not in list(test.FID): 
        fid_train_not_in_fid_test.append(fid)

# =============================================================================
# Detect which FID2 tiles are not in FID1
# =============================================================================      
fid_test_not_in_fid_train = []
for fid in train.FID:
    if fid not in list(train.FID): 
        fid_test_not_in_fid_train.append(fid)
        
        

# =============================================================================
# makes both dataset equal number of tiles
# =============================================================================
train = train.set_index('FID').drop(fid_train_not_in_fid_test).reset_index()
test  = test.set_index('FID').drop(fid_test_not_in_fid_train).reset_index()

MYLABEL = 'c_start_0'
if sum(train[MYLABEL] - test[MYLABEL]) == 0: 
    print('THE TWO DATAFRAMES ARE THE SAME!')
    
    
index2FID_train = train['FID']
index2FID_test  = test['FID']

columns_to_delete=[
'MAPID', 'lat', 'lon', 'geometry_nwf', 'geometry_neigh','geometry',
'FID', 
        ]
test_mean = train.mean()
test_std  = train.std()


# =============================================================================
# preparing datasets for regression
# =============================================================================
train_norm = train.drop(columns_to_delete, axis=1)
train_norm = (train_norm - train_norm.mean())/train_norm.std()

corr_df = train_norm.astype(float)
corr = train_norm.corr()

row_to_del =[]
col_to_keep=[]
#comment second if line to include start/finel count as regression feature
for c in train.columns:
    if ('sum' in c)   or ('count' in  c)\
    or ('Gi_' in c) :
        row_to_del.append(c)
#
    if 'start' in c or 'final'  in c: col_to_keep.append(c)
corr = corr.loc[corr.columns.difference(row_to_del)][col_to_keep]
##
##
test_norm = test.drop(columns_to_delete, axis=1)
test_norm = (test_norm - test_norm.mean())/test_norm.std()
#
#
#
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats

nof = 10
for tb in range(0,7):
    prediction_label = 'c_start_%d'%tb
    print('tb:',tb)

    y_train = train_norm[prediction_label].values
    df_temp = corr[prediction_label].sort_values(ascending=False).iloc[1:nof+1]
    X_train = train_norm[df_temp.index].astype(float)
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    pvalues = est2.pvalues
    tvalues = est2.tvalues
    
    
    feat_sig = pvalues[pvalues < tvalues]
    print('feat sig:', len(feat_sig))
    print('F:', est2.fvalue)
    
    X_test = test_norm[df_temp.index].astype(float).values
    y_test = test_norm[prediction_label]
    reg = LinearRegression().fit(X_train,y_train)
    y_pred = np.matmul(X_test,reg.coef_)
    
    
    MSE = mean_squared_error(y_test, y_pred)
    RSS = (377)*MSE
    RSE = np.sqrt(RSS/(len(y_test)-nof-1))
    
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    print()
    
    diff = y_pred - y_test
    corr_err = diff.autocorr()


#for tb in range(0,7):
#    for nof in range(20,21,5):
#        prediction_label = 'c_start_%d'%tb
#        y_train = train_norm[prediction_label].values
#        
#        df_temp = corr[prediction_label].sort_values(ascending=False).iloc[1:nof+1]
#        X_train = train_norm[df_temp.index].astype(float)
#        X2 = sm.add_constant(X_train)
#        
#
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
#        est = sm.OLS(y_pred, X2)
#        est2 = est.fit()
#        
#        y_pred =( y_pred * test_std[prediction_label]) + test_mean[prediction_label]
#        y_test =( y_test * test_std[prediction_label]) + test_mean[prediction_label]
#        err =  np.abs((y_test-y_pred)) / y_test
#        
#        break
#    break
#
#
#index_to_test = df_temp.index.tolist()
#
#for label in index_to_test:
#    fig,ax=plt.subplots()
#    ax.scatter(train[prediction_label], 
#               train[label], 
#               marker='x', 
#               s=0.5)
#    ax.set_xlabel(label)
#    ax.set_ylabel(prediction_label)
#    
##
##        
##        '''
##        Plot
##        '''
##        fig, ax = plt.subplots(1,3,figsize=(21,7))
##        ax[0].set_title('Prediction %s nof: %d'%(prediction_label, nof) )
##        ax[0].plot(y_pred, label='pred', linestyle='--')
##        ax[0].plot(y_test.values, label='test', linestyle=':')
##        ax[0].legend()
##
##        
##        ax[1].set_title("Error perc. %s nof: %d"%(prediction_label, nof))
##        print('err perc, media mediana: %.2f %.2f'%( err.mean(), err.median()))
##        ax[1].plot(err)
##        ax[1].set_ylim([0,100])
##        
##        
###            
##    print()
###    
###    
##    
#
##    
##    
##    
##    
##    
##    
