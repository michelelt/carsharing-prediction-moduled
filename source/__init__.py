#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:31:21 2019

@author: mc
"""

import os
def init_dir():
    for mydir in ['data', 'credential']:
        if not os.path.isdir('../%s'%mydir):
            os.mkdir('../%s'%mydir)