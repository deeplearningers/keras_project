#coding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model,save_model,load_model
from keras.layers.core import Dense,Activation, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as processimage

#拉取原始数据
(X_train,Y_train),(X_test,Y_test) = mnist.load_data('F:\\keras_project\\mnist.npz')
print( X_train.shape ,Y_train.shape)
print( X_test.shape,Y_test.shape)

#准备数据
#reshape
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
#设置成浮点型,这样收敛更好
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255#归一化，真正转成小数了
X_test /=255

#设置基础参数
batch_size = 1024#批量
nb_class = 10 #类别，0~9
nb_epochs = 5 #训练次数

#Class vectors [0,0,0,0,0,0,0,1(7),0,0] #转成二进制 one hot
Y_test = np_utils.to_categorical(Y_test,nb_class) #定义LABEL类数量
Y_train = np_utils.to_categorical(Y_train,nb_class)

#设置网络结构
model = Sequential() #实例化序列网络
#1st layer 一层
model.add(Dense(units=1024,input_shape=(784,))) #input_dim=784 注意这里说明输入的是维度? 是一维的 张量
model.add(Activation('relu'))
model.add(Dropout(0.2))# overfit

#2nd layer二层
model.add(Dense(units=256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#3rd layer三层
model.add(Dense(10)) #注意这里必须要是10 因为我们有10个类（0123456789）
model.add(Activation('softmax'))#最后一个用 softmax激活函数

#编译 Compile 
model.compile(
loss = 'categorical_crossentropy',#定义损失函数
optimizer = 'rmsprop', #adam,SGD #定义优化函数
metrics = ['accuracy'], #计算模式，精确
)

#启动网络训练
Trainning = model.fit(
X_train,Y_train, #给网络输入训练样本
batch_size = batch_size,
epochs = nb_epochs,
validation_data = (X_test,Y_test), #这个是测试集,会打印测试集结果
verbose = 2, #训练时候的显示模式
)
save_model(model,'mnist_cnn.h5')