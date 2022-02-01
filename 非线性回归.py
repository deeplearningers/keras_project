import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential#按顺序构成的模型
from keras.layers import Dense,Activation#全连接层

from keras.optimizers import SGD#导入优化器，更改学习率等

#使用numpy生成200个数据
x_data = np.linspace(-0.5,0.5,200)
noise =  np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise#开平方

#随机点显示
plt.scatter(x_data,y_data)
plt.show()

#构建数据顺序模型
model = Sequential()
#1-10-1，添加隐10个隐藏层
model.add(Dense(input_dim=1,units=10,activation = 'tanh'))#输出一维，输入一维
#加入激活函数，不加的话默认是线性的,或者在上边加入激活函数也可,tanh,relu,elu,selu
#model.add(Activation('tanh'))
#添加全连接层
model.add(Dense(input_dim=10,units=1,activation = 'tanh'))#输入是上一层的输出10
#model.add(Activation('tanh'))

#自主定义优化算法
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd,loss='mse')#编译模型，优化器是随机梯度下降法；loss是均方误差；

#训练6001个批次
for step in range(6001):
    cost = model.train_on_batch(x_data,y_data)#每次训练一个批次，一个批次是全部数据
    if step % 500 == 0:
        print('cost: ',cost)#每500个batch打印一次loss值
#打印权值和偏置值
W,b  = model.layers[0].get_weights()
print('W: ',W,'b: ',b)

#x_data 输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)

#显示随机点
plt.scatter(x_data,y_data)
#显示预测结果
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()


