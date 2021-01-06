#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:43:17 2020

@author: nephilim
"""
import numpy as np

def tpowGain(data,twtt,power):
    '''
    Apply a t-power gain to each trace with the given exponent.
    INPUT:
    data      data matrix whose columns contain the traces
    twtt      two-way travel time values for the rows in data
    power     exponent
    OUTPUT:
    newdata   data matrix after t-power gain
    '''
    factor = np.reshape(twtt**(float(power)),(len(twtt),1))
    factmat = np.tile(factor,(1,data.shape[1]))  
    return np.multiply(data,factmat)


# aa=tpowGain(Profile,np.arange(1500)/4,2.5)
# pyplot.figure(1)
# pyplot.imshow(aa,vmin=np.min(aa),vmax=np.max(aa),extent=(0,1,0,1),cmap=cm.seismic)
# pyplot.figure(2)
# pyplot.imshow(Profile,vmin=np.min(Profile),vmax=np.max(Profile),extent=(0,1,0,1),cmap=cm.seismic)