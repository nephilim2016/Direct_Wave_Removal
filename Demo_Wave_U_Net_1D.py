#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:23:52 2020

@author: nephilim
"""

import numpy as np
import DataNormalized
import Wave_U_Net_1D
from pathlib import Path
import skimage.transform
from matplotlib import pyplot
import os
import T_PowerGain


def Load_Data(dir_path):
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_original=[]
    data_segment=[]
    for idx in range(file_num):
        for idx_ in range(2):
            data_ori_=np.load(dir_path+'/%s_iter_record_%s_comp.npy'%(idx,idx_))
            data_ori=np.zeros((1600,172))
            data_ori[:1500,:]=data_ori_
            data_ori=T_PowerGain.tpowGain(data_ori,np.arange(1600)/4,0)
            data=DataNormalized.DataNormalized(data_ori)
            mm,nn=data.shape
            direct1D=np.sum(data,axis=1)/nn
            direct1D[600:None]=direct1D[600]
            direct1D=np.reshape(direct1D,(mm,-1))
            direct2D=np.tile(direct1D,(1,nn))
            NoDirect=data-direct2D
            data_original.append(data)
            data_segment.append([direct2D,NoDirect])
    return data_original,data_segment

def CreateTrainData(data_original,data_segment,signal_size):
    data_ori=np.zeros((len(data_original)*data_original[0].shape[1],signal_size,1))
    data_L=np.zeros((len(data_original)*data_original[0].shape[1],signal_size,1))
    iter_=0
    for idx in range(len(data_original)):
        for idx_num in range(data_original[idx].shape[1]):
            data_ori[iter_,:,0]=data_original[idx][:,idx_num]
            data_L[iter_,:,0]=data_segment[idx][0][:,idx_num]
            iter_+=1
    return data_ori,data_L
    
if __name__=='__main__':
    dir_path='./DataFile'
    data_original,data_segment=Load_Data(dir_path)
    signal_size=1600
    data_ori,data_L=CreateTrainData(data_original,data_segment,signal_size)
    # idx=8
    # data=data_original[idx]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,1,0,1),vmin=0.8*np.min(data),vmax=0.8*np.max(data))
    
    # data=data_segment[idx][1]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,1,0,1),vmin=0.8*np.min(data),vmax=0.8*np.max(data))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    FilePath='./SeismicModel'
    epochs=50
    WaveUNet=Wave_U_Net_1D.Wave_U_Net_1D(signal_shape=(1600,1))
    WaveUNet_model=WaveUNet.Build_Wave_U_Net_1D()
    
    inputs_train=data_ori[2000:3000]
    outputs_train=data_L[2000:3000]
    inputs_validation=data_ori[:1000]
    outputs_validation=data_L[:1000]
    save_path_name='./Wave_U_Net_1D'
    batch_size=64
    history,test_loss,Model=WaveUNet.Train_Wave_U_Net_1D(WaveUNet_model,epochs,batch_size,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    Predict_Data=WaveUNet_model.predict(data_ori[70:90])
    
    idx_=1
    aa1=Predict_Data[idx_,:,0]
    cc1=data_ori[idx_,:,0]
    
    pyplot.figure()
    pyplot.plot(aa1)

    pyplot.figure()
    pyplot.plot(cc1-aa1)
    
    pyplot.figure()
    pyplot.plot(aa1)
    pyplot.plot(cc1)