#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:27:07 2019

@author: mc
"""

import geopandas as gpd
import pandas as pd
import  matplotlib.pyplot as plt
import random

def add_emergencies_column(is_train):
    def rnd_color():
        
        
        r =  lambda: random.randint(0,255)
        color_str = '#%02X%02X%02X' % (r(),r(),r())
        return color_str
    
    
    crs_ = {'init': 'epsg:4326'}
    

    
    path = '../data/Vancouver/Opendata/'
    data = 'shape_city_streets/public_streets.shp'
    
    gdf = gpd.read_file(path+data).to_crs(crs_)
    
    # =============================================================================
    # createion of the figure jouning same index streets
    # =============================================================================
    street_names_double = gdf.groupby('HBLOCK').count()['USE']
    street_names_double = street_names_double[street_names_double >= 2].sort_values(ascending=False).reset_index()
    
    gdf_reduced = gdf.set_index('HBLOCK').loc[street_names_double.iloc[0]['HBLOCK']].reset_index()
    my_hblock='GRANVILLE BRIDGE'
    streets = gdf.dissolve(by='HBLOCK',  aggfunc='sum').reset_index()
    
    data = 'street_intersections_shp/street_intersections.shp'
    gdf =  gpd.read_file(path+data).to_crs(crs_)
    a = gdf.groupby('XSTREET').count()
    intersections = gdf\
            .dissolve(by='XSTREET', aggfunc='sum')\
            .reset_index()\
            .drop('ATSTREET', axis=1)\
            .rename(columns={'XSTREET':'HBLOCK'})
    intersections['HBLOCK']  = intersections\
                                .apply(lambda x: 'INTERSECTION '+ x.HBLOCK, 
                                       axis=1)
    
    final_df = streets.append(
            intersections,
            ignore_index=True
            )
    
    
    # =============================================================================
    # standarzie the 311 calss dataset in order to merge with streets
    # =============================================================================
    path = '../data/Vancouver/Opendata/CaseLocationsDetails_2017_CSV/201710CaseLocationsDetails.csv'
    
    df = pd.read_csv(path)
    
    
    with open('date_limits.txt', 'r') as fp:
        dates = fp.readlines()

    import datetime
    limits = {}
    for date in dates:
        date = date.split('#')
                
                          
        year    = date[1].split(' ')[0].split('-')[0]
        month   = date[1].split(' ')[0].split('-')[1]
        day     = date[1].split(' ')[0].split('-')[2]
        hour    = date[1].split(' ')[1].split(':')[0]
        minutes = date[1].split(' ')[1].split(':')[1]
        seconds = date[1].split(' ')[1].split(':')[2]
    
        dt = datetime.datetime(int(year), int(month),   int(day), 
                               int(hour), int(minutes), int(seconds))
        
        limits[date[0]] = dt
    
    path = '../data/Vancouver/Opendata/CaseLocationsDetails_2017_CSV/201710CaseLocationsDetails.csv'
    df = pd.read_csv(path)
    df['date'] =pd.DatetimeIndex( df['Year'].map(str)  +\
                 '-' +df['Month'].map(str) +\
                 '-' +df['Day'].map(str) +\
                 ' ' +df['Hour'].map(str) +\
                 ':' +df['Minute'].map(str) )

    if is_train == True:
        df = df[df.date <= limits['f_date_train']]
        print('len df', len(df))
    else:
         df = df[df.date > limits['i_date_test']]
         
    def merge_hblock_street_name(hb, sn):
        hb = hb.replace('#', '0')
        new_str = '%s %s'%(hb, sn.rstrip())
        new_str = new_str.replace(' - ', '-')
        new_str = new_str.replace('0-0 ', '')
        if new_str[0:2]  == '00':
            new_str  = new_str.replace('00', '0')
            
        
        if '-' in new_str.split(' ')[0]:
            codes = new_str.split(' ')[0]
            codeA, codeB, = int(codes.split('-')[0]),int(codes.split('-')[1]) + 1
            corrected_str  = "%d-%d"%(codeA, codeB)
            new_str  = new_str.replace(
                    new_str[:len(corrected_str)],
                    corrected_str
                    )
        
        return new_str.upper()
    
    df['Street_Name'] = df['Street_Name'].str.lstrip()
    df['Street_Name_2'] = df\
        .apply(lambda x: merge_hblock_street_name(x.Hundred_Block, x.Street_Name),
               axis=1)
    df = df.set_index('Street_Name_2')
    df = df.groupby('Street_Name_2').count()['Year']
    
    joined = final_df\
                .set_index('HBLOCK')\
                .join(df)\
                .fillna(0)
    print(joined.Year.sum())
                
    # =============================================================================
    # merge with train dataset
    # =============================================================================
    
    train = pd.read_csv('../data/Vancouver/Regression/train.csv')
    tiles = gpd.read_file('../data/Vancouver/tiles_metric_None_None').to_crs(crs_).set_index('FID')
    tiles = tiles.loc[train.FID]
    train.set_index('FID', inplace=True)
    
    train_gdf = gpd.GeoDataFrame(train, crs=crs_, geometry=tiles.geometry)
    
    train_gdf_merged = gpd.sjoin(train_gdf, joined, op='intersects').reset_index()
    emeregency_per_zone = train_gdf_merged.groupby('index').sum()['Year'].fillna(0)
    train['Emergencies'] = emeregency_per_zone
    train_gdf['Emergencies'] = emeregency_per_zone
    
    train['Emergencies'] = train['Emergencies'].fillna(0)
    if is_train == True:
        train.to_csv('../data/Vancouver/Regression/train_emer.csv')
    else:
        train.to_csv('../data/Vancouver/Regression/test_emer.csv')
    
    print(emeregency_per_zone.sum())
    


add_emergencies_column(is_train=True)
add_emergencies_column(is_train=False)

             
             
