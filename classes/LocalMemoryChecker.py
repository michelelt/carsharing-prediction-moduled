#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:21:27 2019

@author: mc
"""
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from .FileNameCreator import FileNameCreator



class LocalMemoryChecker:
    
    def __init__(self, city, i_date, f_date, data_path):
        self.city = city
        self.data_path = data_path
        self.i_date = i_date
        self.f_date = f_date
        self.fnc = FileNameCreator(self.i_date, self.f_date,  self.city)
        
    def isDatasetDownloaded(self,file):
#        filename = self.fnc.create_name(file)
        path = '%s%s/%s_%s'%(self.data_path, 
                                            self.city, 
                                            self.city, 
                                            file )
#        
        
        if file == 'tiles':
            
            for ext in ['cpg', 'dbf', 'prj', 'shp', 'shx']:
                to_test = path
                to_test = to_test + '/%s_tiles.%s'%(self.city, ext)
#                print(to_test)
                if os.path.isfile(to_test) ==  False:
                    return False
#                
        if file == 'tiles_metric':
            path = path + '_'  + self.fnc.create_dir('')[-39:] + '/'
            print (path)

            for ext in ['cpg', 'dbf', 'prj', 'shp', 'shx']:
                to_test = path
                to_test = to_test + '/%s_tiles_metric.%s'%(self.city, ext)
                print(to_test)
                if os.path.isfile(to_test) ==  False:
                    return False
            return True

        return os.path.isfile('%s%s/%s_%s'%(self.data_path, 
                                            self.city, 
                                            self.city, 
                                            file ))
#    

#i_date = datetime.datetime(2017, 9, 6, 0, 0, 0)
#f_date = datetime.datetime(2017, 9, 26, 1, 0, 0)  
#city = 'Vancouver'
#data_path = './../data/'
#lmc = LocalMemoryChecker(city,i_date, f_date, data_path)
#lmc.isDatasetDownloaded('filtered_binned_merged')
###        
#zz = data_path+\
#                               city+\
#                               '/%s_filtered_binned_merged.csv/'%city
#
#if lmc.isDatasetDownloaded('filtered_binned_merged.csv'):
#    print ('8888')
#    df = pd.read_csv(data_path+\
#                               city+\
#                               '/%s_filtered_binned_merged.csv'%city,
#                               )