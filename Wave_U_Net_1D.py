#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:33:26 2020

@author: nephilim
"""

import keras

class Wave_U_Net_1D():
    def __init__(self,signal_shape):
        self.__name__='Wave U Net 1D'
        self.signal_shape=signal_shape
    
    
    def Build_Wave_U_Net_1D(self):        
        input_signal=keras.layers.Input(shape=self.signal_shape,name='InputSignal')
        x1=keras.layers.Conv1D(filters=64,kernel_size=15,strides=1,padding='same')(input_signal)
        x1=keras.layers.LeakyReLU(alpha=0.1)(x1)
        # x1=keras.layers.ReLU()(x1)
        x1=keras.layers.BatchNormalization(momentum=0.8)(x1)
        
        x2=keras.layers.Conv1D(filters=128,kernel_size=7,strides=1,padding='same')(x1)
        x2=keras.layers.MaxPooling1D()(x2)
        x2=keras.layers.LeakyReLU(alpha=0.1)(x2)
        # x2=keras.layers.ReLU()(x2)
        x2=keras.layers.BatchNormalization(momentum=0.8)(x2)
        
        x3=keras.layers.Conv1D(filters=256,kernel_size=5,strides=1,padding='same')(x2)
        x3=keras.layers.MaxPooling1D()(x3)
        x3=keras.layers.LeakyReLU(alpha=0.1)(x3)
        # x3=keras.layers.ReLU()(x3)       
        x3=keras.layers.BatchNormalization(momentum=0.8)(x3)
        
        x4=keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,padding='same')(x3)
        x4=keras.layers.MaxPooling1D()(x4)
        x4=keras.layers.LeakyReLU(alpha=0.1)(x4)
        # x4=keras.layers.ReLU()(x4)
        x4=keras.layers.BatchNormalization(momentum=0.8)(x4)
        
        x5=keras.layers.Conv1D(filters=1024,kernel_size=3,strides=1,padding='same')(x4)
        x5=keras.layers.MaxPooling1D()(x5)
        x5=keras.layers.LeakyReLU(alpha=0.1)(x5)
        # x5=keras.layers.ReLU()(x5)
        x5=keras.layers.BatchNormalization(momentum=0.8)(x5)
        
        x6=keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,padding='same')(x5)
        x6=keras.layers.UpSampling1D(size=2)(x6)
        x6=keras.layers.LeakyReLU(alpha=0.1)(x6)
        # x6=keras.layers.ReLU()(x6)
        x6=keras.layers.BatchNormalization(momentum=0.8)(x6)
        
        x_=keras.layers.concatenate([x6,x4],axis=2)
        x7=keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,padding='same')(x_)
        x7=keras.layers.UpSampling1D(size=2)(x7)
        x7=keras.layers.LeakyReLU(alpha=0.1)(x7)
        # x7=keras.layers.ReLU()(x7)
        x7=keras.layers.BatchNormalization(momentum=0.8)(x7)
        
        x_=keras.layers.concatenate([x7,x3],axis=2)
        x8=keras.layers.Conv1D(filters=128,kernel_size=7,strides=1,padding='same')(x_)
        x8=keras.layers.UpSampling1D(size=2)(x8)
        x8=keras.layers.LeakyReLU(alpha=0.1)(x8)
        # x8=keras.layers.ReLU()(x8)
        x8=keras.layers.BatchNormalization(momentum=0.8)(x8)
        
        x_=keras.layers.concatenate([x8,x2],axis=2)
        x9=keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding='same')(x_)
        x9=keras.layers.UpSampling1D(size=2)(x9)
        x9=keras.layers.LeakyReLU(alpha=0.1)(x9)
        # x9=keras.layers.ReLU()(x9)
        x9=keras.layers.BatchNormalization(momentum=0.8)(x9)
        
        x_=keras.layers.concatenate([x9,x1],axis=2)
        x10=keras.layers.Conv1D(filters=1,kernel_size=5,strides=1,padding='same')(x_)
        x10=keras.layers.LeakyReLU(alpha=0.1)(x10)
        # x10=keras.layers.ReLU()(x10)
        x10=keras.layers.BatchNormalization(momentum=0.8)(x10)
        
        x_=keras.layers.concatenate([x10,input_signal],axis=2)
        output_signal=keras.layers.Conv1D(filters=1,kernel_size=1,strides=1,padding='same',activation='sigmoid')(x_)
        
        model=keras.models.Model(inputs=input_signal,outputs=output_signal)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        return model
    
    def Train_Wave_U_Net_1D(self,model,epochs,batch_size,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                        keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=batch_size,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_Wave_U_Net_1D(self,WeightsPath):
        input_signal=keras.layers.Input(shape=self.signal_shape,name='InputSignal')
        x1=keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,padding='same')(input_signal)
        x1=keras.layers.LeakyReLU(alpha=0.3)(x1)
        
        x2=keras.layers.Conv1D(filters=128,kernel_size=3,strides=2,padding='same')(x1)
        x2=keras.layers.LeakyReLU(alpha=0.3)(x2)
        
        x3=keras.layers.Conv1D(filters=256,kernel_size=3,strides=2,padding='same')(x2)
        x3=keras.layers.LeakyReLU(alpha=0.3)(x3)
        
        x4=keras.layers.Conv1D(filters=512,kernel_size=3,strides=2,padding='same')(x3)
        x4=keras.layers.LeakyReLU(alpha=0.3)(x4)
        
        x5=keras.layers.Conv1D(filters=1024,kernel_size=3,strides=2,padding='same')(x4)
        x5=keras.layers.LeakyReLU(alpha=0.3)(x5)
        
        x6=keras.layers.Conv1D(filters=512,kernel_size=3,strides=1,padding='same')(x5)
        x6=keras.layers.UpSampling1D(size=2)(x6)
        x6=keras.layers.LeakyReLU(alpha=0.3)(x6)
        
        x_=keras.layers.concatenate([x6,x4],axis=2)
        x7=keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,padding='same')(x_)
        x7=keras.layers.UpSampling1D(size=2)(x7)
        x7=keras.layers.LeakyReLU(alpha=0.3)(x7)
        
        x_=keras.layers.concatenate([x7,x3],axis=3)
        x8=keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,padding='same')(x_)
        x7=keras.layers.UpSampling1D(size=2)(x8)
        x8=keras.layers.LeakyReLU(alpha=0.3)(x8)
        
        x_=keras.layers.concatenate([x8,x2],axis=3)
        x9=keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,padding='same')(x_)
        x9=keras.layers.UpSampling1D(size=2)(x9)
        x9=keras.layers.LeakyReLU(alpha=0.3)(x9)
        
        
        output_signal=keras.layers.Conv1D(filters=1,kernel_size=1,strides=1,padding='same',activation='sigmoid')(x9)
        
        model=keras.models.Model(inputs=input_signal,outputs=output_signal)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='mse')
        model.load_weights(WeightsPath)
        return model