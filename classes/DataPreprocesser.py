#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:24:37 2019

@author: mc
"""

import os
import pandas as pd

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint


import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from GlobalsFunctions import haversine, crs_
from .DataDownloader import DataDownloader

class DataPreprocesser:
    
    
    def __init__(self, city, data_path, *args, **kwargs):
        self.city = city
        self.provider  = 'car2go'
        self.data_path = data_path
        
        self.i_date = kwargs.get('i_date', None)
        self.f_date = kwargs.get('f_date', None)
        self.is_downloaded = False
        
    def upload_bookigns(self):
        if not os.path.isfile(self.data_path+self.city+'/%s_raw.csv'%self.city):
            print('Download data')
            dd = DataDownloader(self.city, self.provider)
            
            self.booking = dd.query_data(dd.booking_collection, 
                                         i_date = self.i_date,
                                         f_date = self.f_date)
            self.booking.to_csv(self.data_path+self.city+'/%s_raw.csv'%self.city, 
                                index=False)
            self.is_downloaded = True
        else:
            print('Upload data from local')
            self.booking  = pd.read_csv(self.data_path+self.city+'/%s_raw.csv'%self.city)
            self.is_downloaded = True
            
            
    def filter_time(self, min_time, max_time):
        self.booking= self.booking[( self.booking.duration >= min_time)
                                    &self.booking.duration <= max_time]
        return
    
    
    
    def filter_distance(self, distance):
        self.booking = self.booking[(self.booking.distance >= distance)]
        return
        
    
        
    def detect_spatial_outliers(self):
        
        if len(self.booking[self.booking.columns[0]])> 1e5:
            df_test = self.booking.sample(n=int(self.booking.shape[0]*0.05), random_state=1)
        else:
            df_test = self.booking
        
        print('DBSCAN on %d elements' %len(df_test))
        
        fig,ax = plt.subplots()
        ax.scatter(df_test.start_lon, df_test.start_lat, s=0.5, color='red')
        
        coords = df_test[['start_lat', 'start_lon']].values
        kms_per_radian = 6371.0088
        epsilon = 1 / kms_per_radian
        db = DBSCAN(eps=epsilon, 
                    min_samples=100, 
                    algorithm='ball_tree', 
                    metric='haversine').fit(np.radians(coords))
        
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        
        print('Cluser IDs:', set(cluster_labels))
        print('Number of clusters: {}'.format(num_clusters))
        
        
        '''
        Attach the label to each booking
        '''
        df_test['cluster'] = cluster_labels
        
        '''
        retrieve extremes of the squares
        '''
        clean_df = df_test[df_test.cluster >=  0]
        outli_df = df_test[df_test.cluster == -1]
        
        fig,ax = plt.subplots()
        ax.set_title('from dbscan')
        ax.scatter(clean_df.start_lon, clean_df.start_lat, s=0.5, color='green')
        ax.scatter(outli_df.start_lon, outli_df.start_lat, s=0.5, color='red')
        
        
        
        
        self.min_lat = max(clean_df.start_lat.min(), clean_df.end_lat.min())
        self.min_lon = max(clean_df.start_lon.min(), clean_df.end_lon.min())
        
        self.max_lat = min(clean_df.start_lat.max(), clean_df.end_lat.max())
        self.max_lon = min(clean_df.start_lon.max(), clean_df.end_lon.max())

        return df_test

    
    def filter_spatial_outlier(self):
        self.detect_spatial_outliers()
        self.booking = self.booking[
                  (self.booking.start_lat >= self.min_lat)
                & (self.booking.start_lon >= self.min_lon)
                & (self.booking.start_lat <= self.max_lat)
                & (self.booking.start_lon <= self.max_lon)
                
                & (self.booking.end_lat >= self.min_lat)
                & (self.booking.end_lon >= self.min_lon)
                & (self.booking.end_lat <= self.max_lat)
                & (self.booking.end_lon <= self.max_lon)
                ]
        
    def standard_filtering(self):
        if not os.path.isfile(self.data_path+self.city+'/%s_filtered_binned.csv'%self.city):
            print('Init L:%d '%len(self.booking) )
            
            print('Filter time')
            self.filter_time(60, 3600)
            print('L:%d\n'%len(self.booking ))

            
            
            print('Filter distances')
            print('L:%d\n'%len(self.booking ))
            self.filter_distance(700)
            
            print('Filter spatial outliers')
            print('L:%d\n'%len(self.booking ))
            self.filter_spatial_outlier()
            
            print('Set time bins')
            print('L:%d\n'%len(self.booking ))
            self.set_timebin()
            
#            fig,ax = plt.subplots()
#            ax.scatter(self.booking.start_lon, self.booking.start_lat, s=0.5)
            
            print('save')
#            self.booking.to_csv(self.data_path+\
#                                self.city+\
#                                '/%s_filtered_binned.csv'%self.city, 
#                                index=False)
            
            
            
        else:
            print('Dataset already preprocessed')
            self.booking = pd.read_csv(self.data_path+self.city+'/%s_filtered_binned.csv' %self.city)
        return

    def set_timebin(self):
        
        self.booking['time_bin'] = -1
        time_bins = [[ 1, 2, 3, 4, 5, 6],
                     [ 7, 8, 9],
                     [10,11,12],
                     [13,14,15],
                     [16,17,18],
                     [19,20,21],
                     [22,23,0]
                     ]
        time_bin_val = 0
        for time_bin in time_bins:
        #    print (time_bin)
            self.booking.loc[
                    self.booking[self.booking.Hour.isin(time_bin)].index,
                    'time_bin'] = time_bin_val
            time_bin_val+=1

        return
    
    
        
    


#dp = DataPreprocesser('Toronto', './../data/')
#dp.upload_bookigns()
#bookings_before_filtering = dp.booking
#dp.standard_filtering()
#bookings_after_filtering = dp.booking
#dp.set_timebin()
#bookings_after_timebinning = dp.booking
#bookings_after_timebinning.to_csv('../data/Torino/Torino_filtered.csv')
##


