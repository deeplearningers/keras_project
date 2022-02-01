#Resnet50--特征提取器，提取sharedmap
#5个stage  5种参数不同的卷积阶段
#计算参数大约2000万个；
#包括卷积块和恒等快-》然后relu-》输入到下一个block
#rennet头部预处理部分maxpool-》中间主副结构,block12不同-》尾巴

import keras.layers as KL
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import numpy as np
import os
from keras.datasets import mnist

def building_block(filters,block):
    #block类型判断
    if block !=0:
        stride = 1#简单结构
    else:
        stride = 2#复杂结构
    def f(x):
        #主通路结构
        y = KL.Conv2D(filters=filters,kernel_size=(1,1),strides=stride)(x)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(filters=filters, kernel_size=(3,3), padding='same')(y)
        y = KL.BatchNormalization(axis=3)(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(filters=4*filters,kernel_size=(1,1))(y)
        y = KL.BatchNormalization(axis=3)(y)
        #副路
        if block == 0:
            shortcut = KL.Conv2D(filters=filters*4,kernel_size=(1,1),strides=stride)(x)
            shortcut = KL.BatchNormalization()(shortcut)
        else:
            shortcut = x
        #相加
        y = KL.Add()([y,shortcut])
        y = KL.Activation('relu')(y)
        return y
    return f

def ResNet_Extractor1(Xtrain,Ytrain,Xtest,Ytest):
    #头部 输入
    input = KL.Input([28,28,1])
    x = KL.Conv2D(filters=64,kernel_size=(3,3),padding='same')(input)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.Activation('relu')(x)

    #分配stage  block关系
    filters = 64
    block = [2,3,3]#控制stage数量
    for stage,block_num in enumerate(block):
        #print('-------stage----',stage,'---')
        for block_id in range(block_num):
            #print('--block--',block_id,'----')
            x = building_block(filters=filters,block=block_id)(x)
        filters *=2#每个stage中filters加倍

    #尾部 输出
    x = KL.AveragePooling2D(pool_size=(2,2))(x)
    x = KL.Flatten()(x)
    x = KL.Dense(units=10,activation='softmax')(x)

    model = Model(inputs = input,outputs =x)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics =['acurracy'])
    history = model.fit(Xtrain,Ytrain,epochs=6,batch_size =200,verbose =1,validation_data=(Xtest,Ytest))
    model.save('resnetMnist.h5')
    return model

def ResNet_Extractor(inputs):
    #头部 输入
    x = KL.Conv2D(filters=64,kernel_size=(3,3),padding='same')(inputs)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.Activation('relu')(x)

    #分配stage  block关系
    filters = 64
    block = [2,2]#控制stage数量
    for stage,block_num in enumerate(block):
        #print('-------stage----',stage,'---')
        for block_id in range(block_num):
            #print('--block--',block_id,'----')
            x = building_block(filters=filters,block=block_id)(x)
        filters *=2#每个stage中filters加倍
    return x


#shared map和anchor提取
def RpnNet(featuremap,k=9):
    #共享
    sharedMap = KL.Conv2D(filters=256,kernel_size=(3,3),padding ='same')(featuremap)
    sharedMap = KL.Activation('linear')(sharedMap)
    #计算rpn分类前后景
    rpn_classifcation = KL.Conv2D(filters=2*k,kernel_size=(1,1))(sharedMap)
    rpn_classifcation = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,2]))(rpn_classifcation)
    rpn_classifcation = KL.Activation('linear',name='rpn_classification')(rpn_classifcation)

    rpn_probaility = KL.Activation('softmax',name='rpn_probility')(rpn_classifcation)
    #计算回归修正
    rpn_position =  KL.Conv2D(filters=4*k,kernel_size=(1,1))(sharedMap)
    rpn_position = KL.Activation('linear')(rpn_position)
    rpn_BoundingBox = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,4]),name='rpn_POS')(rpn_position)
    return rpn_classifcation,rpn_probaility,rpn_BoundingBox

def main1():
    (Xtrain,Ytrain) ,(Xtest,Ytest)= mnist.load_data()

    Xtrain = Xtrain.reshape(-1,28,28,1)
    Xtest = Xtest.reshape(-1,28,28,1)
    Xtrain = Xtrain/255.0
    Xtest = Xtest/255.0

    Ytrain = np_utils.to_categorical(Ytrain,10)
    Ytest = np_utils.to_categorical(Ytest,10)
    ResNet_Extractor1(Xtrain,Ytrain,Xtest,Ytest)

def main():
    x = KL.Input((100,100,3))
    featureMap = ResNet_Extractor(x)
    rpn_classifcation, rpn_probaility, rpn_BoundingBox = RpnNet(featureMap,k=9)
    model = Model(inputs = [x],outputs = [rpn_classifcation, rpn_probaility, rpn_BoundingBox])
    plot_model(model = model,to_file='withsharedmap.png',show_shapes=True)

main()
