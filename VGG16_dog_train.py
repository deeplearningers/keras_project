from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import numpy as np
import json
import warnings
from keras.utils.vis_utils import plot_model#绘制网络结构
from keras.callbacks import TensorBoard
warnings.filterwarnings("ignore")

batch_size = 16
train_data = './dog_data/train/'
test_data = './dog_data/test/'
image_w = 150
image_h = 150

vgg16_model = VGG16(weights='imagenet',include_top=False,
                    input_shape=(image_w,image_h,3))

# 搭建全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(units=256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(units=10,activation='softmax'))#10个类别
model = Sequential()
model.add(vgg16_model)
model.add(top_model)
#绘制网络结构
plot_model(model,to_file='./vgg16_dog.png',show_shapes=True,show_layer_names=True,rankdir='TB')

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
    train_data,
    target_size=(image_w,image_h),
    batch_size=batch_size,
    )

# 测试数据
test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(image_w,image_h),
    batch_size=batch_size,
    )

label = train_generator.class_indices#字典,先键后值
print(label)
label = dict(zip(label.values(), label.keys()))#改称先值后键
with open('label_dog.json','w',encoding='utf-8') as f:
    json.dump(label, f)  # 保存到json文件
    print(label)

# 定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer=SGD(lr=1e-3,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
#训练模型
model.fit_generator(train_generator,steps_per_epoch=len(train_generator),epochs=50,
                    validation_data=test_generator,validation_steps=len(test_generator),
                    callbacks=[TensorBoard(log_dir='./log')],verbose=2)

model.save('model_vgg16_dog.h5')



