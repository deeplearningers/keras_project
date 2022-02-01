#读取样本

import xml.etree.ElementTree as ET
import glob

def parse_label(xml_file):
    #建立一个实例
    tree  = ET.parse(xml_file)
    #建立根目录
    root = tree.getroot()

    width = root.find('size').find('width').text
    height = root.find('size').find('height').text
    image_name = root.find('filename').text
    #数据列表
    category = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for object in root.findall('object'):
        for x in object.iter('name'):
            category.append(x.text)
            xmax.append(object.find('bndbox').find('xmax').text)
            ymax.append(object.find('bndbox').find('ymax').text)
            xmin.append(object.find('bndbox').find('xmin').text)
            ymin.append(object.find('bndbox').find('ymin').text)

    #列表组合
    ground_true_box = [list(box) for box in zip(xmin,ymin,xmax,ymax)]

    return image_name,(width,height),category,ground_true_box

def main():
    for name in glob.glob('F:\\keras_project\\faster_rcnn\\test\\*'):
        print(parse_label(name))
main()


import pandas as pd

list_table = []
for i in range(10,20):
    date = {'name':i+1,"age":i}
    list_table.append(date)
print(list_table)

date_frame = pd.DataFrame(data=list_table,columns= ['age','name'])
date_frame.to_csv('321.csv',index=False,mode ='a',header=True)