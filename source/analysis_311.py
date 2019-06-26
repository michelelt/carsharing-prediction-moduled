#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:34:39 2019

@author: mc
"""

import pandas as pd
import seaborn as sns

path = '../data/Vancouver/Opendata/CaseLocationsDetails_2017_CSV/201710CaseLocationsDetails.csv'

df = pd.read_csv(path)
emergency = df.groupby(['Hour']).count()['Year']

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
streets_name_em =  df.Street_Name_2.unique().tolist()






