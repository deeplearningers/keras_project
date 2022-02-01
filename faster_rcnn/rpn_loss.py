#Resnet50--特征提取器，提取sharedmap
#5个stage  5种参数不同的卷积阶段
#计算参数大约2000万个；
#包括卷积块和恒等快-》然后relu-》输入到下一个block
#rennet头部预处理部分maxpool-》中间主副结构,block12不同-》尾巴

import keras.layers as KL
from keras.models import Model
import keras.backend as K
import keras
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def building_block(filters,block):
    #block类型判断1或2
    if block !=0:
        stride = 1#简单结构
    else:
        stride = 2#复杂结构，两倍下采样
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
        y = KL.Activation('relu',name='last'+str(random.randint(100,300)))(y)
        return y
    return f

#resnet主输入函数
def ResNet_Extractor(inputs):
    #头部 输入
    x = KL.Conv2D(filters=64,kernel_size=(3,3),padding='same')(inputs)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.Activation('relu')(x)

    #控制调用网络结构feature map特征图
    #分配stage  block关系，每个stage有不同的block12数量，
    filters = 64
    block = [2,2,2]#控制stage数量
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

def RPNClassLoss(rpn_match,rpn_Cal):
    rpn_match = tf.squeeze(rpn_match,axis = -1)

    indices = tf.where(K.not_equal(x=rpn_match,y=0))
    #1=1  0,-1 = 0
    anchor_class = K.cast(K.equal(x=rpn_match,y=1),tf.int32)#return 01000
    #原始样本的结果
    anchor_class = tf.gather_nd(params=anchor_class,indices = indices)
    #rpn计算的值
    rpn_cal_class = tf.gather_nd(params=rpn_Cal,indices=indices)
    #loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_cal_class,from_logits=True)
    #判断loss是否正常
    loss = K.switch(condition=tf.size(loss)>0,then_expression=K.mean(loss),
                    else_expression=tf.constant(0.0))
    return loss

#小工具提取
def batch_pack(x,counts,num_rows):
    output=[]
    for i in range(num_rows):
        output.append(x[i,:counts[i]])
    return tf.concat(output,axis=0)

#位置loss
def RpnBBoxLoss(target_bbox,rpn_match,rpn_bbox):
    rpn_match = tf.squeeze(input=rpn_match,axis=-1)
    indexs = tf.where(K.equal(x=rpn_match,y=1)) #正样本位置

    rpn_bbox=tf.gather_nd(params=rpn_bbox,indices=indexs)#rpn预测值

    batch_counts =K.sum(K.cast(K.equal(x = rpn_match,y=1),tf.int32),axis=-1)
    target_bbox = batch_pack(x =target_bbox,counts=batch_counts,num_rows=10)
    #loss计算
    diff = K.abs(target_bbox-rpn_bbox)
    less_than_one = K.cast(K.less(x =diff,y=1.0),tf.float32)
    loss = less_than_one * 0.5*diff**2 + (1-less_than_one)*(diff-0.55)
    loss = K.switch(condition=tf.size(loss) > 0, then_expression=K.mean(loss),
                    else_expression=tf.constant(0.0))
    return loss

#确定input
input_image = KL.Input(shape=[64,64,3],dtype=tf.float32)
input_bbox = KL.Input(shape=[None,4],dtype=tf.float32)
input_class_ids = KL.Input(shape=[None],dtype=tf.int32)
input_rpn_match = KL.Input(shape=[None,1],dtype=tf.int32)
input_rpn_bbox= KL.Input(shape=[None,4],dtype=tf.float32)

#in out put
feature_map = ResNet_Extractor(input_image)
rpn_classifcation,rpn_probaility,rpn_BoundingBox = RpnNet(feature_map,k=9)
loss_rpn_class = KL.Lambda(lambda x:RPNClassLoss(*x),name='classloss')([input_rpn_match,rpn_classifcation])
loss_rpn_bbox =KL.Lambda(lambda x:RpnBBoxLoss(*x),name='bboxloss')([input_rpn_bbox,input_rpn_match,rpn_BoundingBox])

model = Model(inputs=[input_image,input_bbox,input_class_ids,input_rpn_match,input_rpn_bbox],
              outputs = [rpn_classifcation,rpn_probaility,rpn_BoundingBox,loss_rpn_class,loss_rpn_bbox])

#自定义loss 输入
loss_layer1 = model.get_layer('classloss').output
loss_layer2 = model.get_layer('bboxloss').output

model.add_loss(tf.reduce_mean(loss_layer1))
model.add_loss(tf.reduce_mean(loss_layer2))

model.compile(loss=[None]*len(model.output),optimizer=keras.optimizers.SGD(lr=0.00003))

model.summary()

def main():
    x = KL.Input((64,64,3))
    featureMap = ResNet_Extractor(x)
    rpn_classifcation, rpn_probaility, rpn_BoundingBox = RpnNet(featureMap,k=9)
    model = Model(inputs = [x],outputs = [rpn_classifcation, rpn_probaility, rpn_BoundingBox])
    model.summary()
    plot_model(model = model,to_file='loss.png',show_shapes=True)

