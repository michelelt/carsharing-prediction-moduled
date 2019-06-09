#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:53:16 2019

@author: mc
"""

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from GlobalsFunctions import haversine, crs_


from shapely.geometry import MultiPoint, Polygon, Point
import pandas as pd
import geopandas as gpd

class TilesMapCreator:
    
    def __init__(self,df):
        self.df = df
        self.min_lat = min(self.df.start_lat.min(), self.df.end_lat.min())
        self.min_lon = min(self.df.start_lon.min(), self.df.end_lon.min())
        self.max_lat = max(self.df.start_lat.max(), self.df.end_lat.max())
        self.max_lon = max(self.df.start_lon.max(), self.df.end_lon.max())
        self.crs = crs_
        
        
    def find_steps(self, side_length, epsilon):
        step_lon = 0.01
        epsilon = epsilon
        dist = 1000
        step = 0
        while 1:
            if dist >= side_length-epsilon and dist <=side_length + epsilon:
                break
            
            elif dist < 500-epsilon:
                step_lon += step_lon*0.5
            else:
                step_lon -= step_lon*0.5
                
            dist = haversine(self.min_lon, self.min_lat, 
                             self.min_lon+step_lon, self.min_lat)
            step+=1
        
        print ('step_lon found in %d steps'%step)
            
        step_lat = 0.01
        epsilon = epsilon
        dist = 1000
        step = 0
        while 1:
            if dist >= side_length-epsilon and dist <=side_length + epsilon:
                break
            
            elif dist < 500-epsilon:
                step_lat += step_lat*0.5
            else:
                step_lat -= step_lat*0.5
                
            dist = haversine(self.min_lon, self.min_lat, 
                             self.min_lon, self.min_lat+step_lat)
            step+=1
            
        print ('step_lat found in %d steps'%step)
        self.step_lon = step_lon
        self.step_lat = step_lat
    
        return step_lon, step_lat, step
    
    
    def squareize(self, lon_c, lat_c, step_lon, step_lat):
        A = (lon_c, lat_c)
        B = (lon_c + step_lon, lat_c)
        C = (lon_c + step_lon, lat_c + step_lat)
        D = (lon_c, lat_c + step_lat)
            
        return Polygon([A,B,C,D])    
    
    
    def create_empity_tiles_map(self, side_legth, epsilon):
    
        # =========================================================================
        # create new tiles map
        # =========================================================================
        self.find_steps(side_legth, epsilon)
        
        geometry = []
        lat = self.min_lat
        while lat <= self.max_lat + self.step_lat:
            lon = self.min_lon
            while lon <= self.max_lon + self.step_lon:
                geometry.append(self.squareize(lon, lat, self.step_lon, self.step_lat))
                lon += self.step_lon
            lat += self.step_lat
        tiles = gpd.GeoDataFrame(geometry=geometry, crs=self.crs)
        self.tiles = tiles
        return tiles
    
        
