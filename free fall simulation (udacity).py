# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:09:29 2016

@author: Henrik
"""

import numpy as np
import matplotlib.pyplot as plt

def forward_euler(steps, stepsize):
    g = 9.81 #m/s^2
    h = stepsize #s
    
    t = np.zeros(steps+1)
    x = np.zeros(steps+1)
    v = np.zeros(steps+1)
    
    for i in range(steps-1):
        t[i + 1] = i+1
        x[i + 1] = x[i] + h*v[i]
        v[i + 1] = v[i] + h*g
    return t,x,v
    

    