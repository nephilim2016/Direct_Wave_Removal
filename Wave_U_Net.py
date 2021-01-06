#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:41:01 2020

@author: nephilim
"""

import keras

class Wave_U_Net():
    def __init__(self,image_shape):
        self.__name__='Wave U Net'
        self.image_shape=image_shape
    def Build_Wave_U_Net(self):
        input_image=keras.layers.Input(shape=self.image_shape,name='InputImage')
        x1=keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',activation='relu')(input_image)
        x2=keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x1)
        x3=keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x2)
        x4=keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x3)
        
        x5=keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x4)
        
        x6=keras.layers.Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding='same')(x5)
        x6=keras.layers.Activation('relu')(x6)
        
        x_=keras.layers.concatenate([x6,x4],axis=3)
        x7=keras.layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2,padding='same')(x_)
        x7=keras.layers.Activation('relu')(x7)
        x_=keras.layers.concatenate([x7,x3],axis=3)
        x8=keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='same')(x_)
        x8=keras.layers.Activation('relu')(x8)
        x_=keras.layers.concatenate([x8,x2],axis=3)
        x9=keras.layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=2,padding='same')(x_)
        x9=keras.layers.Activation('relu')(x9)
        
        output_image_L=keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=1,padding='same',activation='sigmoid')(x9)
        output_image_S=keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=1,padding='same',activation='sigmoid')(x9)
        model=keras.models.Model(inputs=input_image,outputs=[output_image_L,output_image_S])
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss=['mse','mse'])
        return model
    
    def Train_Wave_U_Net(self,model,epochs,batch_size,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                        keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=batch_size,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_Wave_U_Net(self,WeightsPath):
        input_image=keras.layers.Input(shape=self.image_shape,name='InputImage')
        x1=keras.layers.Conv2D(filters=16,kernel_size=(11,11),strides=1,padding='same',activation='relu')(input_image)
        x2=keras.layers.Conv2D(filters=16,kernel_size=(9,9),strides=2,padding='same',activation='relu')(x1)
        x3=keras.layers.Conv2D(filters=16,kernel_size=(7,7),strides=2,padding='same',activation='relu')(x2)
        x4=keras.layers.Conv2D(filters=16,kernel_size=(5,5),strides=2,padding='same',activation='relu')(x3)
        
        x5=keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x4)
        
        x6=keras.layers.Conv2DTranspose(filters=16,kernel_size=(5,5),strides=2,padding='same')(x5)
        x6=keras.layers.Activation('relu')(x6)
        
        x_=keras.layers.Concatenate()([x6,x4])
        x7=keras.layers.Conv2DTranspose(filters=16,kernel_size=(7,7),strides=2,padding='same')(x_)
        x7=keras.layers.Activation('relu')(x7)
        x_=keras.layers.Concatenate()([x7,x3])
        x8=keras.layers.Conv2DTranspose(filters=16,kernel_size=(9,9),strides=2,padding='same')(x_)
        x8=keras.layers.Activation('relu')(x8)
        x_=keras.layers.Concatenate()([x8,x2])
        x9=keras.layers.Conv2DTranspose(filters=16,kernel_size=(11,11),strides=2,padding='same')(x_)
        x9=keras.layers.Activation('relu')(x9)
        
        output_image_L=keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=1,padding='same',activation='sigmoid')(x9)
        output_image_S=keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=1,padding='same',activation='sigmoid')(x9)
        model=keras.models.Model(inputs=input_image,outputs=[output_image_L,output_image_S])
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        model.load_weights(WeightsPath)
        return model