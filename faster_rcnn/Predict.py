#coding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as processimage

model = load_model('mnist_cnn.h5')

(X_train,Y_train),(X_test,Y_test) = mnist.load_data('F:\\keras_project\\mnist.npz')
testrun = X_test[9999].reshape(1,784)
testlabel = Y_test[9999]
print(testrun.shape,testlabel)
mypred = model.predict(testrun) #调用predict函数预测
print(mypred)
print([myfinal.argmax() for myfinal in mypred])

#用自己的图预测一下
# target_img = processimage.imread('路径+文件名') #读取图片
# target_img = target_img.reshape(1,784) #图片reshape到网络可以接收的维度
# target_img = np.array (target_img) #将图片转numpy数组
# target_img = target_img.astype('float32') #数组转浮点
# target_img /=255 #归一化
#mypred = model.predict(testrun) #调用predict函数预测
#print([myfinal.argmax() for myfinal in mypred]) #使用argmax输出最终结果