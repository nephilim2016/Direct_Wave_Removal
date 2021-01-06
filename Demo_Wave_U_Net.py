#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:26:50 2020

@author: nephilim
"""

import numpy as np
import DataNormalized
import Wave_U_Net
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
            data_ori=np.load(dir_path+'/%s_iter_record_%s_comp.npy'%(idx,idx_))
            data_ori=T_PowerGain.tpowGain(data_ori,np.arange(1500)/4,0)
            data_ori=skimage.transform.resize(data_ori,(256,256),mode='symmetric')
            data=DataNormalized.DataNormalized(data_ori)
            mm,nn=data.shape
            direct1D=np.sum(data,axis=1)/nn
            direct1D[70:None]=direct1D[70]
            direct1D=np.reshape(direct1D,(mm,-1))
            direct2D=np.tile(direct1D,(1,nn))
            NoDirect=data-direct2D
            data_original.append(data)
            data_segment.append([direct2D,NoDirect])
    return data_original,data_segment

def CreateTrainData(data_original,data_segment,image_size):
    data_ori=np.zeros((len(data_original),image_size,image_size,1))
    data_L=np.zeros((len(data_original),image_size,image_size,1))
    data_S=np.zeros((len(data_original),image_size,image_size,1))
    for idx in range(len(data_original)):
        data_ori[idx,:,:,0]=data_original[idx]
        data_L[idx,:,:,0]=data_segment[idx][0]
        data_S[idx,:,:,0]=data_segment[idx][1]
    return data_ori,data_L,data_S
    
if __name__=='__main__':
    dir_path='./DataFile'
    data_original,data_segment=Load_Data(dir_path)
    image_size=256
    data_ori,data_L,data_S=CreateTrainData(data_original,data_segment,image_size)
    # idx=8
    # data=data_original[idx]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,1,0,1),vmin=0.8*np.min(data),vmax=0.8*np.max(data))
    
    # data=data_segment[idx][1]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,1,0,1),vmin=0.8*np.min(data),vmax=0.8*np.max(data))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    FilePath='./SeismicModel'
    ImageShape=(256,256)
    epochs=100
    WaveUNet=Wave_U_Net.Wave_U_Net(image_shape=(256,256,1))
    WaveUNet_model=WaveUNet.Build_Wave_U_Net()
    
    inputs_train=data_ori
    outputs_train=[data_L,data_S]
    inputs_validation=data_ori[:5]
    outputs_validation=[data_L[:5],data_S[:5]]
    save_path_name='./Wave_U_Net'
    batch_size=2
    history,test_loss,Model=WaveUNet.Train_Wave_U_Net(WaveUNet_model,epochs,batch_size,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    Predict_Data=WaveUNet_model.predict(data_ori[0:3])
    
    idx_=2
    aa=Predict_Data[0]
    aa1=aa[idx_,:,:,0]
    bb=Predict_Data[1]
    bb1=bb[idx_,:,:,0]
    pyplot.figure()
    pyplot.imshow(aa1)
    pyplot.figure()
    pyplot.imshow(bb1)
    cc=data_ori[idx_]
    cc1=cc[:,:,0]
    pyplot.figure()
    pyplot.imshow(cc1-bb1)
    