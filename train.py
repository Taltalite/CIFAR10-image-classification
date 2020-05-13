import preprocess
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense
from keras.models import Model
import os
import argparse

def train(config):
    x_img_train_normalize, y_label_train_onehot, x_img_test_normalize, y_label_test_onehot = data_preprocess()
    model = model_gen(config)
    print(model.summary())

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    train_history = model.fit(x_img_train_normalize, y_label_train_onehot,
                              epochs=config.epochs, batch_size=config.batch_size, verbose=2)

def model_gen(config):
    # This returns a tensor
    inputs = Input(shape=(32, 32, 3), dtype='float32', name='inputs')

    layer_conv1 = Conv2D(filters=32, kernel_size=(3, 3),
                         input_shape=(32, 32, 3),
                         activation='relu',
                         padding='same',
                         data_format='channels_last')(inputs)

    layer_dropout1 = Dropout(rate=config.drop_rate)(layer_conv1)

    # 32*32 --> 16*16
    layer_maxpooling1 = MaxPooling2D(pool_size=(2, 2))(layer_dropout1)

    layer_conv2 = Conv2D(filters=64, kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         data_format='channels_last')(layer_maxpooling1)

    layer_dropout2 = Dropout(rate=config.drop_rate)(layer_conv2)

    # 16*16 --> 8*8
    layer_maxpooling2 = MaxPooling2D(pool_size=(2, 2))(layer_dropout2)

    layer_flatten = Flatten()(layer_maxpooling2)

    output = Dense(512, activation='relu')(layer_flatten)

    layer_dropout3 = Dropout(rate=config.drop_rate)(output)

    predict = Dense(10, activation='softmax')(layer_dropout3)

    model = Model(inputs=inputs, outputs=predict)

    return model


def data_preprocess():
    print("start data loading...")
    x_img_train, y_label_train, x_img_test, y_label_test = preprocess.data_load()
    print("complete data loading...\n")
    print("start img normalizing...")
    x_img_train_normalize, x_img_test_normalize = preprocess.img_normalize(x_img_train, x_img_test)
    print("complete img normalizing...\n")
    print("start label onehotkey...")
    y_label_train_OneHot, y_label_test_OneHot = preprocess.label_onehotkey(y_label_train, y_label_test)
    print("complete label onehotkey...\n")

    return x_img_train_normalize, y_label_train_OneHot, x_img_test_normalize, y_label_test_OneHot





if __name__ == "__main__":
    np.random.seed(10)
    parser = argparse.ArgumentParser(description="config setting")
    parser.add_argument("-e", "--epochs", type=int, help="训练轮数")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-d", "--drop_rate", type=float, default=0.25, help="drop rate, 默认是0.25")
    args = parser.parse_args()
    print("config:")
    print(args)
    train(args)



