#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img = cv2.imread(os.path.expanduser('F:\\jupyterDir\\21dl\\SSDMobileNet\\waterdry3\\VOC2007\\JPEGImages\\1_0.jpg'))
img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
img = img.astype(float)
img /= 255
img = np.array([img])

# 初始化TensorFlow的session
with tf.Session() as sess:
    # 读取得到的pb文件加载模型
    with gfile.FastGFile("F:\\keras_project\\ssd_keras\\train_dir\\33.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # 把图加到session中
        tf.import_graph_def(graph_def, name='')

    # 获取当前计算图
    graph = tf.get_default_graph()
    # 从图中获输出那一层
    pred = graph.get_tensor_by_name("output_1:0")
    # 运行并预测输入的img
    res = sess.run(pred, feed_dict={"input_1:0": img})
    # 执行得到结果
    pred_index = res[0][0]
    #打印
    print(pred_index)