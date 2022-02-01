from keras.models import Model
from keras.layers import Lambda,Activation, Dense,Conv2D,Input,BatchNormalization,MaxPooling2D,Flatten
import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model
#定义数据结构
input_tensor_1 = Input([64,64,3])
input_tensor_2 = Input([4,])
input_target = Input([2,])
#网络结构
#第一组
x = BatchNormalization(axis=-1)(input_tensor_1)

x = Conv2D(filters = 32,kernel_size =(3,3),padding ='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(filters = 32,kernel_size =(3,3),padding ='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(units=16)(x)#全连接层
out2 = Dense(units=2)(x)

#第二组
y = Dense(units=32)(input_tensor_2)
out1 = Dense(units=2)(y)
#第三组
z = Dense(units=8)(input_target)
out3 = Dense(units=2)(z)
#自定义loss
def cus_loss1(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))

def cus_loss2(y_true,y_pred):
    return K.mean(K.abs(y_true - y_pred))
#lambda层
loss1 = Lambda(lambda x:cus_loss1(*x),name = 'loss1')([out2,out1])
loss2 = Lambda(lambda x:cus_loss2(*x),name='loss2')([out2,out3])

model = Model(inputs = [input_tensor_1,input_tensor_2,input_target],outputs=[out1,out2,out3,loss1,loss2])
#获取loss
loss_layer1 = model.get_layer('loss1').output
loss_layer2 = model.get_layer('loss2').output
#向模型添加losss
model.add_loss(loss_layer1)
model.add_loss(loss_layer2)

#None表示不需要梯度返回
model.compile(optimizer='sgd',loss=[None,None,None,None,None])


#dataset
def data_gen(number):
    for i in range(number):
        yield [np.random.normal(1,1,size=[1,64,64,3]),
               np.random.normal(1, 1, size=[1,4]),
               np.random.normal(1, 1, size=[1,2])],[]

dataset = data_gen(1000)
#训练
train = model.fit_generator(dataset,epochs=10,steps_per_epoch=20)

plot_model(model=model,to_file ='model.png',show_shapes= True)



