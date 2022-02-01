import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
#from keras.callbacks import TensorBoard
from keras.models import save_model,load_model
from keras.utils.vis_utils import plot_model#绘制网络结构

#创建顺序模型
model = Sequential()
#卷积层1
model.add(Convolution2D(
    input_shape=(28,28,1),#输入尺寸
    filters=32,#卷积核个数
    kernel_size=5,#卷积窗口大小
    strides=1,#步长
    padding='same',#padding方式 same/valid
    activation='relu'#激活函数
))
#池化层1
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
#卷积层2
model.add(Convolution2D(
    #input_shape=(28,28,1),自动计算输入尺寸，不需要填写了
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
#池化层2
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
#将第二个池化层输出扁平化为一维
model.add(Flatten())
#全连接层1
model.add(Dense(units=1024,activation='relu'))#输出1024个神经元，输入默认；
#Dropout
model.add(Dropout(0.5))
#全连接层2
model.add(Dense(units=10,activation='softmax'))

#绘制模型
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False,rankdir='TB')
plt.figure(figsize=(10,10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis('off')
plt.show()