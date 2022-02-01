import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential#按顺序构成的模型
from keras.layers import Dense #全连接层

#使用numpy生成100个随机点
x_data= np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)#扰动,高斯分布
y_data = x_data*0.1 + 0.2 +noise

#随机点显示
plt.scatter(x_data,y_data)
plt.show()

#构建数据顺序模型
model = Sequential()
#在模型中添加全连接层
model.add(Dense(input_dim=1,units=1))#输出一维，输入一维
model.compile(optimizer='sgd',loss='mse')#编译模型，优化器是随机梯度下降SGD，默认学习率很小0.01；loss是均方误差；

#训练3001个批次
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


