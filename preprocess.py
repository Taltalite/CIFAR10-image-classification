label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

import numpy
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils

np.random.seed(10)

# 导入数据集，如果没有就会自动下载
def data_load():
    (x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
    print('train:', len(x_img_train))
    print('test :', len(x_img_test))
    print('train_image :', x_img_train.shape)
    print('train_label :', y_label_train.shape)
    print('test_image :', x_img_test.shape)
    print('test_label :', y_label_test.shape)
    return x_img_train, y_label_train, x_img_test, y_label_test


# 将rgb值正规化
def img_normalize(x_img_train, x_img_test):
    # print(x_img_train[0][0][0])  # （50000，32，32，3）
    x_img_train_normalize = x_img_train.astype('float32') / 255.0
    x_img_test_normalize = x_img_test.astype('float32') / 255.0
    # print(x_img_train_normalize[0][0][0])
    return x_img_train_normalize, x_img_test_normalize


# 将label标签变成one got key
def label_onehotkey(y_label_train, y_label_test):
    y_label_train_OneHot = np_utils.to_categorical(y_label_train)
    y_label_test_OneHot = np_utils.to_categorical(y_label_test)
    # print(y_label_train_OneHot.shape)
    # print(y_label_train_OneHot[:5])
    return y_label_train_OneHot, y_label_test_OneHot

