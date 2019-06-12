#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:54:13 2019

@author: mc
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt


X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
bre
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
print(reg.coef_)
print(reg.intercept_ )
reg.predict(np.array([[3, 5]]))
