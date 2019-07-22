#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:03:43 2019

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

def create_csv_from_metric(metric):
    csv_str = 'var,value\n'
    input_str = str(metric)
    input_str = input_str.split('\n')[0:-1]
    for line in input_str:
        csv_str += line.split(' ')[0]+','+ line.split(' ')[-1]+'\n'
    return csv_str

data_path = './../../data/'
city = 'Vancouver'
ind_var = 'space'

from normalize_neighs import normalize_dataset

'''
cretae a dataset having 21 test neighbours and 1 test (#21)
implenet automatic k-fold validation
'''
train_norm, test_norm, index2FID_train, index2FID_test,\
test_mean, test_std  = normalize_dataset(data_path, city, ind_variable=ind_var)


'''
Run the MRMR feature selection and saving results in 
root/MicheleRankings/outputs/
'''
#from remote_feature_selection import MRMR
#starts_labels = ['c_start_%d'%tb for tb in range(0,7)]
#finals_labels = ['c_final_%d'%tb for tb in range(0,7)]
#for label in starts_labels: 
#    MRMR(train_norm, label)
#    break
#for label in finals_labels: MRMR(train_norm, label)

label = 'c_start_0'
ranked_feat_path = '../../MicheleRankings/outputs/%s/%s_mrmr_regression.csv'%(label, label)
MID_ranks = pd.read_csv(ranked_feat_path)

fp = open(ranked_feat_path, 'r')
lines = fp.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    if i == 0:
        lines[i].insert(1, 'Feature')
    for j in range(len(lines[i])):
        lines[i][j] = lines[i][j].lstrip().rstrip()
        
MID_ranks_2 = pd.DataFrame(lines)

#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
#import statsmodels.api as sm
#from scipy import stats
#
#
##with open('../regression_param_outputs.txt', 'r') as fp:
##    perf_columns = fp.read().lstrip().rstrip().split('\n')
#
#nof = 107
#perf_df = pd.DataFrame()
#s = pd.Series()
#
#for nof in range(1, 107):
#    for src in ['start', 'final']:
#        for tb in range(0,7):
#            prediction_label = 'c_%s_%d'%(src, tb)
#            
#            y_train = train_norm[prediction_label].values
#            df_temp = corr[prediction_label].sort_values(ascending=False).iloc[1:nof+1]
#            X_train = train_norm[df_temp.index].astype(float)
#            X2 = sm.add_constant(X_train)
#            est = sm.OLS(y_train, X2)
#            est2 = est.fit()
#            pvalues = est2.pvalues
#            tvalues = est2.tvalues
#            
#            
#            feat_sig = pvalues[pvalues < tvalues]
#            
#            
#            X_test = test_norm[df_temp.index].astype(float).values
#            y_test = test_norm[prediction_label]
#            reg = LinearRegression().fit(X_train,y_train)
#            y_pred = np.matmul(X_test,reg.coef_)
#            
#            
#            MSE = mean_squared_error(y_test, y_pred)
#            RSS = len(y_test)*MSE
#            RSE = np.sqrt(RSS/(len(y_test) - nof - 1))
#                        
#            diff = y_pred - y_test
#            corr_err = diff.autocorr()
#            
##            print('tb:',tb)
##            print('feat sig:', len(feat_sig))
##            print('F:', est2.fvalue)
##            print('RSE:', RSE)            
##            print('Variance score: %.2f' % r2_score(y_test, y_pred))
##            print(corr_err)
##            print()
#            
#            s['Label'] = prediction_label
#            s['NOF'] =  nof
#            s['MSE'] = MSE
#            s['RSS'] = RSS
#            s['RSE'] = RSE
#            s['F-value'] = est2.fvalue
#            s['F-statistic'] = str(est2.summary()).split('\n')[5].split(' ')[-1] 
#            s['P-values'] =create_csv_from_metric(pvalues)
#            s['T-values'] = create_csv_from_metric(tvalues)
#            s['R2'] = r2_score(y_test, y_pred)
#            s['autocorr_err'] = corr_err
#            s['feat sig'] = len(feat_sig)
#            s['y_pred'] = pd.Series(index=index2FID_test.values, data=y_pred).mul(test_std[prediction_label]).add(test_mean[prediction_label]).to_csv()
#            s['y_test'] = pd.Series(index=index2FID_test.values, data=y_test
#                                                                     .values).mul(test_std[prediction_label]).add(test_mean[prediction_label]).to_csv()
#            s['y_pred_mean'] = y_pred.mean()
#            s['y_test_mean'] = test_mean[prediction_label]
#            s['y_test_std']  = test_std[prediction_label]
#        
#            
#            #abs(pred - test)
#            y_pred_s = pd.Series(y_pred)
#            s['err_abs_rescaled'] = abs(
#                    sum(y_pred_s.mul(test_std[prediction_label]) -\
#                          y_test.mul(test_std[prediction_label])    
#                        )
#                    )
#                    
#                    
#            s['tot_bookings'] = sum(y_test.mul(test_std[prediction_label]).add(test_mean[prediction_label])  )
##            s['err_perc'] = abs(s['err_abs_rescaled']-s['tot_bookings'])*100/s['tot_bookings']
#            
#            perf_df = perf_df.append(s, ignore_index=True)
#
#           
#
#
#perf_df.to_csv('../../data/Vancouver/Regression/Vancouver_perf_%s.csv'%ind_var)
#zzz = s['y_pred']

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
