import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.callbacks import TensorBoard
from keras.models import save_model,load_model
from keras.regularizers import l2#正则化

#载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data('F:\\keras_project\\mnist.npz')
#打印数据格式
print('x_shape: ',x_train.shape)#6000-28-28
print('y_shape: ',y_train.shape)#60000
#60000-28-28->60000-28-28-1（加入图像深度，四维）并归一化处理
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0
#转换label成one hot形式
y_train = np_utils.to_categorical(y_train,num_classes=10)#keras函数，10个分类
y_test = np_utils.to_categorical(y_test,num_classes=10)

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

#保存模型参数和结构
save_model(model,'mnist_cnn.h5')

#只保存模型的参数
model.save_weights('mnist_weihht.h5')
#保存网络结构，载入网络结构
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
# print(json_string)



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