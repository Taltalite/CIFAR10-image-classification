# -*- coding:utf-8 -*-


# 如果没有cifar10数据集，那么运行该.py，会下载数据集到:
# r"C:\Users\<用户名>\.keras\datasets"

import numpy
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
# 导入数据集，如果没有就会自动下载
(x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data()
print('train:',len(x_img_train))
print('test :',len(x_img_test))
print('train_image :',x_img_train.shape)
print('train_label :',y_label_train.shape)
print('test_image :',x_img_test.shape)
print('test_label :',y_label_test.shape)

