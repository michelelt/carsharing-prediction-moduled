#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:10:24 2019

@author: mc
"""

import datetime 
class FileNameCreator:
    
    def __init__(self, i_date, f_date, city):
        self.i_date = str(i_date).replace(' ', 'T').replace(':','-')
        self.f_date = str(f_date).replace(' ', 'T').replace(':','-')
        self.city = city
        
        
    def create_name(self, file_name):
        self.out_file_name ='%s_%s_%s_%s.csv'%(self.city, file_name, self.i_date, self.f_date)
        return self.out_file_name
    
    def create_dir(self, file_name):
        self.out_file_name ='%s_%s_%s'%(file_name, self.i_date, self.f_date)
        return self.out_file_name

#    
#
#i_date = datetime.datetime(2017, 9, 6, 0, 0, 0)
#f_date = datetime.datetime(2017, 9, 6, 1, 0, 0)
#fnc = FileNameCreator(i_date, f_date, 'Vancouver')
#zzz = fnc.create_dir('filtered_binned_merged')
##         