#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:13:38 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
import T_PowerGain

def shrink(X,tau):
    return np.sign(X)*np.maximum((np.abs(X)-tau),np.zeros(X.shape))

def svd_threshold(X,tau):
    U,S,V=np.linalg.svd(X,full_matrices=False)
    return np.dot(U,np.dot(np.diag(shrink(S,tau)),V))

def RobustPCA(M,tau,mu,tol=1e-7,max_iter=1000):
    Lk=Sk=Yk=np.zeros_like(M)
    error=np.inf
    iter_=0
    while error>tol and iter_<max_iter:
        Lk=svd_threshold(M-Sk+1/mu*Yk,1/mu)
        Sk=shrink(M-Lk+1/mu*Yk,tau/mu)
        Yk=Yk+mu*(M-Lk-Sk)
        error=np.linalg.norm(M-Lk-Sk,ord='fro')
        iter_+=1
    return Lk,Sk

if __name__=='__main__':
    data=np.load('0_iter_record_0_comp.npy')
    data=T_PowerGain.tpowGain(data,np.arange(1500)/4,0)
    tau=1/np.sqrt(np.max(data.shape))/8
    mu=np.prod(data.shape)/(4*np.linalg.norm(data,ord=1))
    tol=1e-7
    max_iter=2000
    L,S=RobustPCA(data,tau,mu,tol,max_iter)
    
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    
    pyplot.figure()
    pyplot.imshow(L,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    
    pyplot.figure()
    pyplot.imshow(S,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    
