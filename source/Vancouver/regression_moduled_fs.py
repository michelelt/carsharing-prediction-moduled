#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:41:21 2019

@author: mc
"""

city = 'Vancouver'
data_path  = './../../data/'

loo = LeaveOneOut()
res = []
#
start = time.time()
reg = Regression(data_path, city, norm=False)
