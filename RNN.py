import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.callbacks import TensorBoard
from keras.models import save_model,load_model#可以load模型再接着训练用
from keras.layers.recurrent import SimpleRNN

#数据长度--一行28个像素
input_size = 28
#序列长度--一共28行
time_steps = 28
#隐藏层cell个数
cell_size = 50

#载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data('F:\\keras_project\\mnist.npz')
#打印数据格式
print('x_shape: ',x_train.shape)#6000-28-28
print('y_shape: ',y_train.shape)#60000
#(60000,784)->#(60000,28,28)转换成这种形式
#(60000,28,28) 归一化
x_train = x_train/255.0
x_test = x_test/255.0
#转换label成one hot形式
y_train = np_utils.to_categorical(y_train,num_classes=10)#keras函数，10个分类
y_test = np_utils.to_categorical(y_test,num_classes=10)

#创建顺序模型
model = Sequential()
#循环神经网络
model.add(SimpleRNN(units=cell_size,#输出
                    input_shape=(time_steps,input_size)))#输入
#输出层
model.add(Dense(units=10,activation='softmax'))

#定义优化器
#sgd = SGD(lr=0.2)
adam = Adam(lr=1e-4)#默认0.001
#定义优化器和loss function,设置训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',
              metrics=['accuracy'])#loss-交叉熵,收敛速度快，精度提升
#fit方法训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2#2代表每个epoch打印一条记录
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

#test accuracy:  0.9791
#train accuracy:  0.99925
#train-accuracy比test-accuracy大的多，过拟合了，添加l2正则化

#添加后：train-accuracy和test-accuracy接近了，但是准确率总体下降
#test accuracy:   0.9738
#train accuracy:  0.9862833333333333

#什么时候加？模型对于数据集比较复杂就加上；如果模型对于数据而言不复杂就不用加；

#更换优化器-Adam一般比SGD效果好，收敛速度也快,但是这个并没有体现？
#test accuracy: 0.9729
#train accuracy:  0.9814666666666667

#用CNN
#test accuracy: 0.9927
#train accuracy:  0.9964833333333334


#用RNN---比较差
#test accuracy: 0.9071
#train accuracy:  0.9041166666666667
