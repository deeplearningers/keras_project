from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils.vis_utils import plot_model#绘制网络结构
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


# 定义模型
model = Sequential()
model.add(Conv2D(input_shape=(150,150,3),filters=32,kernel_size=3,padding='same',activation='relu',name='conv_11'))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',name='conv_12'))
model.add(MaxPool2D(pool_size=2, strides=2,name='pool_11'))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu',name='conv_21'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu',name='conv_22'))
model.add(MaxPool2D(pool_size=2, strides=2,name='pool_21'))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',name='conv_31'))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',name='conv_32'))
model.add(MaxPool2D(pool_size=2, strides=2,name='pool_31'))

model.add(Flatten())
model.add(Dense(64,activation='relu',name='Relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax',name='Softmax'))

#绘制网络结构
plot_model(model,to_file='./model_catdog.png',show_shapes=True,show_layer_names=True,rankdir='TB')
#plt.figure(figsize=(10,10))
# img = plt.imread("model_catdog.png")
# plt.imshow(img)
# plt.axis('off')
# plt.show()


# 定义优化器
adam = Adam(lr=1e-4)
# 定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
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
    './cat_dog/train',#训练集保存位置
    target_size=(150,150),#resize图像大小，自己设定
    batch_size=32,
    )
# 测试数据
test_generator = test_datagen.flow_from_directory(
    './cat_dog/test',
    target_size=(150,150),
    batch_size=32,
    )
#打印分类的标签
print(train_generator.class_indices)
#训练模型，每次epoch训练会生成train_generator的训练集，都会是数据增强后的。
model.fit_generator(train_generator,steps_per_epoch=len(train_generator),epochs=60,callbacks=[TensorBoard(log_dir='./log')],
                    validation_data=test_generator,validation_steps=len(test_generator),verbose=2)#400/32=12.5=13个批次
#保存模型
model.save('model_cnn_catdog.h5')

