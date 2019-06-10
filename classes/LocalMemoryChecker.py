#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:21:27 2019

@author: mc
"""


import os
class LocalMemoryChecker():
    
    def __init__(self, city, data_path):
        self.city = city
        self.data_path = data_path
        
    def isDatasetDownloaded(self,file):
        path = '%s%s/%s_%s'%(self.data_path, 
                                            self.city, 
                                            self.city, 
                                            file )
        if file == 'tiles':
            
            for ext in ['cpg', 'dbf', 'prj', 'shp', 'shx']:
                to_test = path
                to_test = to_test + '/%s_tiles.%s'%(self.city, ext)
#                print(to_test)
                if os.path.isfile(to_test) ==  False:
                    return False
                
        if file == 'tiles_metric':
            
            for ext in ['cpg', 'dbf', 'prj', 'shp', 'shx']:
                to_test = path
                to_test = to_test + '/%s_tiles.%s'%(self.city, ext)
#                print(to_test)
                if os.path.isfile(to_test) ==  False:
                    return False
            return True
        
        return os.path.isfile('%s%s/%s_%s'%(self.data_path, 
                                            self.city, 
                                            self.city, 
                                            file ))
    

    
