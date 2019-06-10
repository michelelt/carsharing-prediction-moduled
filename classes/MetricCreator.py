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


from shapely.geometry import MultiPoint, Polygon, Point
import numpy as np
from scipy.spatial.distance import squareform, pdist

from math import *




class MetricCreator:
    
    def __init__(self, bookings, tiles, data_path):
        self.df = bookings            
        self.tiles = tiles
        if 'FID' not in self.tiles.columns:
            self.tiles = self.tiles\
                        .reset_index()\
                        .rename(columns={'index':'FID'})

        
        self.crs = crs_
        self.data_path=data_path
        self.city = bookings.iloc[0]['city']
        self.tiles_with_metric=None
        
        lmc = LocalMemoryChecker(self.city, self.data_path)
        if lmc.isDatasetDownloaded('tiles_metric'):
            print('carico da file')
            self.tiles_with_metric = gpd.read_file(self.data_path+\
                                       self.city+\
                                       '/%s_tiles_metric/%s_tiles_metric.shp'%(self.city, self.city),
                                       crs=self.crs)
        
    def merge_tiles_with_bookings(self):
        
        if ('index_start' in list(self.df.columns)) and ('index_end' in list(self.df.columns))\
        or\
        ('FID_right' in list(self.df.columns)) and ('FID_left' in list(self.df.columns)):
            return
    
        for string in ['start','end'] :
            self.df['geometry'] = self.df.apply(lambda x: Point(x[string+"_lon"], 
                                                                x[string+"_lat"]), 
                                                            axis=1)
            df = gpd.GeoDataFrame(self.df, crs=self.crs)
        
            merged = gpd.sjoin(df, self.tiles, how='left', op='within')
            merged = merged.rename(columns={
                    "index_right": "index_"+string, 
                    })

            merged = merged.fillna(-1)
            self.df = merged
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

        #    log = open('log.txt', 'w')
        tiles = self.tiles
#        print (self.tiles.columns)
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
                
    #            log.write('--------------\n')
    #            log.write('closer: ' +str(closer) +'\n')
    #            log.write('n:%d\n' % n)
    #            log.write('x_2_sum:'  + str(x_2_sum) +'\n')
    #            log.write('X:' +str(X) + '\n')
    #            log.write('W:' +str(W) + '\n')
    #            log.write('--------------\n')
                
                if D == 0: Gi[cell] = 0
                else: Gi[cell] = N/D
                
            
            # =====================================================================
            # computing Z-sscore for each Gi index
            # =====================================================================
            neigh['Gi_%d'%i] = pd.DataFrame.from_dict(Gi, orient='index')
    #        neigh['z_Gi_%d'%i] = (neigh['Gi_%d'%i] - neigh['Gi_%d'%i].mean()) / (neigh['Gi_%d'%i].std() )
           
        neigh = gpd.GeoDataFrame(neigh, geometry=cells_gdf.geometry, crs=crs_).reset_index()  
#        print()
#        print(self.tiles.index)
#        print(neigh.index)
#        print()
#        self.tiles = self.tiles.join(neigh.set_index('FID'),n how='left')
#    #    log.close()     
        
        self.tiles = neigh
    
        return neigh
            
        
    def compute_metrics_per_tile(self, save):
        if not self.tiles_with_metric is None:
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
        
        self.compute_Gi()
        self.tiles_with_metric = self.tiles
        
        
        if save:
            if not os.path.isdir(self.data_path+self.city):
                print('Impossivle to save the metrics tabele')
                return self.tiles
            
            #'data_path/Toronto/Toronto_tiles_shp'
            self.tiles.to_file(self.data_path+self.city+'/%s_tiles_metric'%self.city)            
        
        
        return self.tiles_with_metric
        






















