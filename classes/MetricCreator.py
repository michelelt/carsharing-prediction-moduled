#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:25:39 2019

@author: mc
"""

import pandas as pd
import geopandas as gpd




import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from GlobalsFunctions import haversine, crs_

from .DataPreprocesser import DataPreprocesser
from .TilesMapCreator import TilesMapCreator
from .LocalMemoryChecker import LocalMemoryChecker
from .FileNameCreator import FileNameCreator


from shapely.geometry import MultiPoint, Polygon, Point
import numpy as np
from scipy.spatial.distance import squareform, pdist

from math import *




class MetricCreator:
    
    def __init__(self, bookings, tiles, i_date, f_date, data_path):
        self.df = bookings            
        self.tiles = tiles
        if 'FID' not in self.tiles.columns:
            self.tiles = self.tiles\
                        .reset_index()\
                        .rename(columns={'index':'FID'})

        
        self.crs = crs_
        self.data_path=data_path
        self.city = bookings.iloc[0]['city']
        self.tiles_with_metric = None
        self.i_date = i_date
        self.f_date = f_date
        
        lmc = LocalMemoryChecker(self.city, self.i_date, self.f_date, self.data_path)
        
        self.fnc = FileNameCreator(self.i_date, self.f_date, self.city)
        file_name = self.fnc.create_dir('tiles_metric')
        
        if lmc.isDatasetDownloaded(file_name):
            print('Metric uploaded from local')
            self.tiles_with_metric = gpd.read_file(self.data_path+\
                                       self.city+\
                                       '/%s/%s.shp'%(file_name, file_name),
                                       crs=self.crs)
        
        
#        if lmc.isDatasetDownloaded('filtered_binned_merged.csv'):
        fileid='filtered_binned_merged'
        if os.path.isfile(self.data_path + self.city+ '/' + self.fnc.create_name(fileid)):
            print('Merged dataset uploaded from local')
            self.df = pd.read_csv(self.data_path+\
                                       self.city+\
                                       '/' + self.fnc.create_name(fileid)
                                       )
        square_test = tiles.geometry[0]
        if self.is_square(square_test):
            self.tiles_are_squares = True
        else:
            self.tiles_are_squares = False
        # print('Tiles are squares',  self.tiles_are_squares)
        
    def merge_tiles_with_bookings(self, vancouver_olny=False):
        
        if ('index_start' in list(self.df.columns)) and ('index_end' in list(self.df.columns))\
        or\
        ('FID_right' in list(self.df.columns)) and ('FID_left' in list(self.df.columns)):
            print('Dataset alrady merged with map. Uploaded from local.')
            return

        print('Merging bookings with tiles')
    
        for string in ['start','end'] :
            self.df['geometry'] = self.df.apply(lambda x: Point(x[string+"_lon"], 
                                                                x[string+"_lat"]), 
                                                            axis=1)
            df = gpd.GeoDataFrame(self.df, crs=self.crs)
        
            merged = gpd.sjoin(df, self.tiles[['geometry', 'FID']], how='left', op='within')
            merged = merged.rename(columns={
                    "index_right": "index_"+string, 
                    })

            merged = merged.fillna(-1)
            self.df = merged
            fileid = 'filtered_binned_merged'
            file_name = self.fnc.create_name(fileid)
            
        if vancouver_olny == True:
            
            print('len with -1: %d', len(self.df))
            self.df = self.df[self.df.FID_right > -1]
            self.df = self.df[self.df.FID_left > -1]
            print('len without -1: %d', len(self.df))
            self.df.to_csv(self.data_path + '%s/%s'%(self.city, file_name))
        return 
    
    
    
    def create_oparative_area(self):
        try:
            set_areas = np.unique(np.concatenate((self.df.index_start.unique(), 
                                                  self.df.index_end.unique()))
                        )
        
            self.tiles = self.tiles[ (self.tiles.FID.isin(set_areas))
                                    &(self.tiles.FID > -1)  
                                    ]
            
        except KeyError:
            print('Merge bookings with tiles before')
            
        return
            
    def compute_distance_matrix(self):

        if 'lat' not in self.tiles.columns: self.tiles['lat'] = self.tiles.geometry.centroid.y
        if 'lon' not in self.tiles.columns: self.tiles['lon'] = self.tiles.geometry.centroid.x
    
    
        ID_columns='FID'
        distance_matrix = pd.DataFrame(
                squareform(pdist(self.tiles[[ID_columns, 'lat', 'lon']].iloc[:, 1:])), 
                columns=self.tiles[ID_columns].unique(), 
                index=self.tiles[ID_columns].unique()
                )
        self.distance_matrix = distance_matrix
        return
    
    def compute_Gi(self):
        
        self.compute_distance_matrix()
        tiles = self.tiles
        neigh = tiles.set_index('FID')
        cells_gdf = tiles.set_index('FID')
        
        time_bins = self.df.time_bin.unique()
        for i in time_bins:
            label = 'sum_%d'%i
            neigh[label] =  neigh['c_start_%d'%i] - neigh['c_final_%d'%i]
                    
            n = len(neigh.index)
            Gi = {}
                
            for cell in cells_gdf.index:
            
                weights =  self.distance_matrix[cell]
                weights = weights[weights <= 0.0069]
                closer = neigh.loc[weights.sort_values().index]
                
                W = len(closer)
                X = sum(closer[label])
                x_2_sum = sum(np.square(closer[label]))
            
                
                N = sqrt(n-1)* n * X - (sum(closer[label] * W ))
                D = sqrt(n * x_2_sum - X*X)*sqrt(n*W - W*W)
                
                if D == 0: Gi[cell] = 0
                else: Gi[cell] = N/D
                
            
            # =================================================================
            # computing Z-sscore for each Gi index
            # =================================================================
            neigh['Gi_%d'%i] = pd.DataFrame.from_dict(Gi, orient='index')
           
        neigh = gpd.GeoDataFrame(neigh, geometry=cells_gdf.geometry, crs=crs_).reset_index()  
        
        self.tiles = neigh
    
        return neigh

    def is_square(self, square):

        'tested only on Vancouver'
        bounds = square.bounds
        verteces = list(zip(*square.exterior.coords.xy))
        
        if len(verteces) != 4: return False
        
        for i in range(0,4):
    
            l1 = haversine(verteces[i][0],
                           verteces[i][1],
                           verteces[(i+1)%4][0],
                           verteces[(i+1)%4][1],
                           )
            
            l2  = haversine(verteces[(i+1)%4][0],
                       verteces[(i+1)%4][1],
                       verteces[(i+2)%4][0],
                       verteces[(i+2)%4][1],
                       )
            
            if not isclose(l1, l2, abs_tol=3.5):
                return False
        return True
            
        
    def compute_metrics_per_tile(self, save):
        if not self.tiles_with_metric is None:
            print('Upload metrics from local')
            return self.tiles_with_metric

            
        self.create_oparative_area()
    
        try :
            df = pd.DataFrame(index=self.tiles.index)
            df['count_start'] = self.df.groupby('index_start').count()['_id']
            df['count_end'] = self.df.groupby('index_end').count()['_id']
            init = self.df\
                .groupby(['time_bin', 'index_start'])\
                .count()['_id']\
                .reset_index()\
                .set_index('index_start')
                
            final = self.df\
                .groupby(['time_bin', 'index_end'])\
                .count()['_id']\
                .reset_index()\
                .set_index('index_end')
            
            
            time_bins = self.df.time_bin.unique()            
            for tb in time_bins:
                
                df['c_start_%d'%tb] = init[init.time_bin == tb]['_id']
                df['c_final_%d'%tb] = final[final.time_bin == tb]['_id']
        
        except KeyError:
            print('Merge bookings with tiles before')
            return
        
#        self.df = self.df.fillna(0)
        self.tiles  = self.tiles.join(df, how='left')
        self.tiles = self.tiles.fillna(0)
        
        if self.tiles_are_squares:
            print('Gi computed')
            self.compute_Gi()
        else:
            print('Gi not computed')


        self.tiles_with_metric = self.tiles
        
        
        if save:
            if not os.path.isdir(self.data_path+self.city):
                print('Impossivle to save the metrics table')
                return self.tiles

            #'data_path/Toronto/Toronto_tiles_shp'
            fileid = 'tiles_metric'
            print('len tiles ', len(self.tiles))
            file_name = self.fnc.create_dir(fileid) + '_' + str(len(self.tiles))
            print(file_name)
#            self.tiles.to_file(self.data_path+self.city+'/%s'%file_name)            
        
        
        return self.tiles_with_metric
        






















