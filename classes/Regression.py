#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:57:31 2019

@author: mc
"""

import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



class Regression:
    
    def __init__(self, data_path, city, norm):
        self.data_path =data_path
        self.city = city
        train = pd.read_csv('%s%s/Regression/dataset_train_emer.csv'%(data_path, city))
        valid = pd.read_csv('%s%s/Regression/dataset_test_emer.csv'%(data_path, city))
        self.complete_dataset = train.append(valid, ignore_index=True)
        self.norm =  norm
        self.targets = []
        for c in self.complete_dataset.columns:
            if 'c_start' in c or 'c_final' in c:
                self.targets.append(c)
                
        self.targets_df = self.complete_dataset[self.targets]
        self.complete_dataset = self.complete_dataset.drop(self.targets, axis=1)
        
    def rfr_regression(self, n_estimators, random_state):
        
        rfr = RandomForestRegressor(n_estimators=n_estimators,
                                    random_state=random_state)
        
        return rfr
    

    
    def preprocess_data(self):
        
        for c in self.complete_dataset.columns:
            if ('sum' in c) or ('count' in  c)\
            or ('start' in c) or ('final'  in c)\
            or ('Gi_' in c) or ('m_age' in c)\
            or ('f_age' in c) or ('NAME' in  c)\
            or ('MAPID' in c) or ('FID' in c)\
            or ('geometry' in c):
                
                self.complete_dataset.drop(c, axis=1, inplace=True)
#                self.valid.drop(c, axis=1, inplace=True)
                
    def split_datasets(self, target, train_indeces, valid_indeces, *args, **kwargs):
        self.target = target

        '''
        keep all the columns passed as optioanl parameter
        '''
        columns = kwargs.get('features_to_keep', self.complete_dataset.columns)
        self.reduced_dataset = self.complete_dataset[columns]

        self.target = target
        self.train = self.reduced_dataset.loc[train_indeces]
#                        .drop(['MAPID', 'geometry', 'NAME'], axis=1)
        self.valid = self.reduced_dataset.loc[valid_indeces]
#                        .drop(['MAPID', 'geometry', 'NAME'], axis=1)
                        
        self.train[target] = self.targets_df.loc[self.train.index, target]
        self.valid[target] = self.targets_df.loc[self.valid.index, target]
        
        self.train_target = self.train[target]
        self.valid_target = self.valid[target]
        
        self.mean = self.train.mean()
        self.std  = self.train.std() 
        
        if self.norm == True:
            self.train = (self.train - self.mean)/self.std
            self.valid = (self.valid - self.mean)/self.std
            self.train_target = self.train[target]
            self.valid_target = self.valid[target]
        
        self.train.drop(self.target, axis=1, inplace=True)
        self.valid.drop(self.target, axis=1, inplace=True)

            

    def config_svr_regressor(self, kernel, *args, **kwargs ):
        if kernel == 'poly':
            C       = kwargs.get('C', 100)
            gamma   = kwargs.get('gamma', 'auto')
            epsilon = kwargs.get('epsilon', .1)
            degree  = kwargs.get('degree', 2)
            coef0   = kwargs.get('coeff01',1)
            svr = SVR(kernel=kernel, 
                      C=C, 
                      gamma=gamma, 
                      degree=degree, 
                      epsilon=epsilon, 
                      coef0=coef0)
            
        elif kernel == 'linear':
            C       = kwargs.get('C', 100)
            gamma   = kwargs.get('gamma', 'auto')
            epsilon = kwargs.get('epsilon', .1)
            svr= SVR(kernel=kernel, C=C, gamma=gamma)
            
        elif kernel == 'rbf':
            C       = kwargs.get('C', 100)
            gamma   = kwargs.get('gamma', 0.1)
            epsilon = kwargs.get('epsilon', .1)
            svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
            
            
        else:
            svr = None
        
        return svr
        
      
    
        
    def perform_regression(self, algorithm, *args, **kwargs):
        
        s ={}
        if algorithm == 'rfr':
            n_estimators = kwargs.get('n_estimators', 10)
            random_state = kwargs.get('random_state', 0)
            s['n_estimators'] = n_estimators
            s['random_state'] = random_state
            s['algorithm'] = algorithm  
            method = self.rfr_regression(n_estimators, random_state)
            
        
        
        elif algorithm == 'svr':
            kernel = kwargs.get('kernel', 'linear')
            s['algorithm'] = algorithm
            s['kernel'] = kernel
            
            
            method = self.config_svr_regressor(kernel, kwargs)
            if method == None:
                print('SVR %s Kernel not implemented'%kernel)
                return  False
            
        else:
            print('%s method not yet implemented')
            return False
        
        regressor = method.fit(self.train, self.train_target)
        y_pred_train = regressor.predict(self.train)
        er_r_pred_train =  sum(abs(y_pred_train-self.train_target)/self.train_target)\
                            /(len(self.train_target))
        y_pred_valid = regressor.predict(self.valid)

        s['FID_valid'] = self.valid.index.values[0]
        s['y_pred_valid'] =  y_pred_valid[0]
        s['y_valid'] = self.valid_target.values[0]
        s['er_r_pred_train'] = er_r_pred_train
        s['target'] = self.target
        s['mean_target'] =  self.mean[self.target]
        s['std_target'] = self.std[self.target]
        s['is_normed'] = self.norm
        s['nof'] = len(self.reduced_dataset.columns)
        
        if self.norm==True:
            s['rb_y_pred'] =  s['y_pred_valid']*self.std[self.target] + self.mean[self.target]
            s['rb_y_valid'] = s['y_valid'] * self.std[self.target] + self.mean[self.target]
            
            y_pred_train = y_pred_train * self.std[self.target] + self.mean[self.target]
            self.train_target = self.train_target * self.std[self.target] + self.mean[self.target]
            
 
            
            s['rb_er_r_pred_train'] = sum(abs(y_pred_train-self.train_target)/self.train_target)\
                            /(len(self.train_target))
            
            
        if  algorithm == 'rfr':
            score = pd.DataFrame(method.feature_importances_)\
                                .rename(columns={0:'score'})
            features = pd.DataFrame(self.reduced_dataset.columns.tolist())\
                                .rename(columns={0:'feature'})
                                
            s['rank'] = features.join(score).set_index('feature').to_json()     

        self.results = s
        del s
        
        return True


    def set_norm(self, value): self.norm  = value

    
        
        

    
from sklearn.model_selection import LeaveOneOut 
import time
import datetime
  

    
#city = 'Vancouver'
#data_path  = './../data/'
##
#loo = LeaveOneOut()
#res = pd.DataFrame()
#res = []
#
#start = time.time()
#reg = Regression(data_path, city, norm=True)
#reg.preprocess_data()
##    
#start = time.time()
#target = 'c_start_0'
#for kernel in ['rbf']:
#    for train_index, valid_index in loo.split(reg.complete_dataset):
#        
#        reg.set_norm(False)
#        reg.split_datasets(target, train_index, valid_index)
#        train_target, valid_target  = reg.train_target, reg.valid_target
#        reg.perform_regression('svr',  kernel  = kernel)
#        res.append(reg.results)
#        
#        reg.set_norm(True)
#        reg.split_datasets(target, train_index, valid_index)
##            train, valid = reg.train, reg.valid
#        train_target, valid_target  = reg.train_target, reg.valid_target
#
#        reg.perform_regression('svr', kernel=kernel)
#        res.append(reg.results)
#        print()
#            
#            
#            
#end = time.time() - start
#print('Execution done in %s minutes.' %str(datetime.timedelta(seconds=end)))
            
        
    


    
    




    
    