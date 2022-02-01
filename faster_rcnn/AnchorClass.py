import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#原始输入图
Sample_raw_x =128
Sample_raw_y =128
#8倍下采样
rpn_stride = 8

Feature_size_X = Sample_raw_x/rpn_stride
Feature_size_Y = Sample_raw_y/rpn_stride

scales = [1,2,4]#长宽
ratios = [0.5,1,2]#比率

# fx =  np.arange(Feature_size_X)
# fy =  np.arange(Feature_size_Y)
# F_X ,F_Y = np.meshgrid(fx,fy)

#长宽和比率组合
def anchor(Feature_size_X,Feature_size_Y,rpn_stride,scales,ratios):
    #组合尺寸和比例
    scales,ratios = np.meshgrid(scales,ratios)
    scales,ratios = scales.flatten(),ratios.flatten()

    #计算anchor尺寸
    scalesX = scales * np.sqrt(ratios) #宽度
    scalesY = scales / np.sqrt(ratios) #长度

    #anchor point 映射
    ShiftX = np.arange(0,Feature_size_X) * rpn_stride
    ShiftY = np.arange(0,Feature_size_Y) * rpn_stride
    #anchor point 在原图的位置
    ShiftX,ShiftY = np.meshgrid(ShiftX,ShiftY)#x，y是anchor中心点

    #每个anchor点上需要有9个尺寸的anchor框
    centerX,anchorX = np.meshgrid(ShiftX,scalesX)
    centerY,anchorY = np.meshgrid(ShiftY,scalesY)

    #stack 各种尺寸，各种比例 对应各种长度
    anchor_center = np.stack([centerY,centerX],axis =2).reshape(-1,2)
    anchor_size = np.stack([anchorY,anchorX],axis =2).reshape(-1,2)

    #左上 右下 的坐标点输出
    boxes = np.concatenate([anchor_center-0.5*anchor_size,anchor_center+0.5*anchor_size],axis =1)#别落下axis
    return boxes

anchors = anchor(Feature_size_X,Feature_size_Y,rpn_stride,scales,ratios)

plt.figure(figsize=(10,10))
image = Image.open('F:\\keras_project\\faster_rcnn\\test.jpg')
plt.imshow(image)
asx = plt.gca()#get current axs

for i in range(anchors.shape[0]):
    box = anchors[i]
    #print(box)
    rec = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],edgecolor='r',facecolor='none')
    asx.add_patch(rec)
plt.show()

