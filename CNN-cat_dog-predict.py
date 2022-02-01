from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np

label = np.array(['cat','dog'])
# 载入模型
model = load_model('./catdog_vgg16.h5')
# 导入图片
image = load_img('./cat_dog/test/cat/cat.1093.jpg')

image = image.resize((150,150))
image = img_to_array(image)
image = image/255#归一化
image = np.expand_dims(image,0)#扩展维度
print(image.shape)
#预测
print(label[model.predict_classes(image)])