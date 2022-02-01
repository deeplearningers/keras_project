from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils.vis_utils import plot_model#绘制网络结构
from keras.callbacks import TensorBoard
import numpy as np

#不包含top即全连接层，用imagenet训练好的；
vgg16_model = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))

# 自己搭建全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))#取后三个维度
top_model.add(Dense(units=256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(units=2,activation='softmax'))#输出2个神经元
#连接起来新模型
model = Sequential()
model.add(vgg16_model)
model.add(top_model)
#绘制网络结构
plot_model(model,to_file='./vgg16_catdog.png',show_shapes=True,show_layer_names=True,rankdir='TB')
#数据增强
train_datagen = ImageDataGenerator(
    rotation_range = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 20,       # 随机错切变换
    zoom_range = 0.2,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    fill_mode = 'nearest',   # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
)

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    './cat_dog/train',
    target_size=(150,150),
    batch_size=32,
    )
# 测试数据
test_generator = test_datagen.flow_from_directory(
    './cat_dog/test',
    target_size=(150,150),
    batch_size=32,
    )

print(train_generator.class_indices)
#编译模型
model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
#训练模型
model.fit_generator(train_generator,steps_per_epoch=len(train_generator),epochs=50,
                    validation_data=test_generator,validation_steps=len(test_generator),
                    callbacks=[TensorBoard(log_dir='./log')],verbose=2)
#保存模型
model.save('catdog_vgg16.h5')
print('train finished!')