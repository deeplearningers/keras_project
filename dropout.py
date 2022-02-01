import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import save_model,load_model

#载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data('F:\\keras_project\\mnist.npz')
#打印数据格式
print('x_shape: ',x_train.shape)#6000-28-28
print('y_shape: ',y_train.shape)#60000
#60000-28-28->60000-784，并归一化处理
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0
#转换label成one hot形式
y_train = np_utils.to_categorical(y_train,num_classes=10)#keras函数，10个分类
y_test = np_utils.to_categorical(y_test,num_classes=10)
#创建模型，输入784个神经元，输出10个神经元
model = Sequential()
#加上200个神经元的隐藏层
model.add(Dense(input_dim=784,units=200,bias_initializer='one',activation='tanh'))
#加入dropout层防止过拟合
model.add(Dropout(0.4))
#再加上100个神经元的隐藏层
model.add(Dense(input_dim=200,units=100,bias_initializer='one',activation='tanh'))
#偏置值1；激活函数softmax-输出转成概率值
model.add(Dense(input_dim=100,units=10,bias_initializer='one',activation='softmax'))

#定义优化器
sgd = SGD(lr=0.2)
#定义优化器和loss function,设置训练过程中计算准确率
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])#loss-交叉熵,收敛速度快，精度提升
#fit方法训练模型
model.fit(x_train,y_train,batch_size=30,epochs=10,verbose=2#2代表每个epoch打印一条记录
        ,callbacks = [TensorBoard(log_dir='F:\\keras_project\\log')])#日志
#评估模型
loss,accuracy = model.evaluate(x_test,y_test)
print('\ntest loss: ',loss)
print('test accuracy: ',accuracy)

loss1,accuracy1 = model.evaluate(x_train,y_train)
print('\ntrain loss: ',loss1)
print('train accuracy: ',accuracy1)

#保存模型
save_model(model,'mnist.h5')

#test accuracy:  0.9791
#train accuracy:  0.99925
#train-accuracy比test-accuracy大的多，过拟合了，添加dropout

#添加后：train-accuracy和test-accuracy接近了，但是准确率总体下降
#test accuracy:   0.9743
#train accuracy:  0.9856166666666667