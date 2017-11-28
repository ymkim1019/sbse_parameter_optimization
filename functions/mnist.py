from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def f_mnist(params):
    """
    num_of_conv_layers : int, (2 ~ 10)
    conv_kernel_size : int, (3 ~ 10)
    conv_output_size : int, (32 ~ 128)
    conv_dropout_rate : float, (0.1 ~ 0.99)
    maxpooling_size : int, (2 ~ 10)
    num_of_dense_layers : int, (2 ~ 5)
    dense_output_size : int, (32 ~ 128)
    dense_drop_out_rate : float, (0.1 ~ 0.99)
    learning_rate : float, (0.1, 0.01, 0.001, 0.0001)
    :param params: list of parameters
    :type params: list
    :return: fitness
    :rtype: float
    """

    num_of_conv_layers = 3
    conv_kernel_size = 3
    conv_output_size = params[0]
    conv_dropout_rate = params[1]
    maxpooling_size = params[2]
    num_of_dense_layers = params[3]
    dense_output_size = params[4]
    dense_drop_out_rate = params[5]
    learning_rate = params[6]

    batch_size = 128
    num_classes = 10
    epochs = 5

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    # import gzip
    # f = gzip.open('mnist.pkl.gz', 'rb')
    # if sys.version_info < (3,):
    #     data = pickle.load(f)
    # else:
    #     data = pickle.load(f, encoding='bytes')
    # f.close()
    # (x_train, y_train), (x_test, y_test) = data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(conv_kernel_size, conv_kernel_size),
                     activation='relu',
                     input_shape=input_shape))
    for i in range(num_of_conv_layers-1):
        model.add(Conv2D(conv_output_size, (conv_kernel_size, conv_kernel_size), activation='relu'))
        model.add(MaxPooling2D(pool_size=(maxpooling_size, maxpooling_size)))
        model.add(Dropout(conv_dropout_rate))
    model.add(Flatten())
    for i in range(num_of_dense_layers - 1):
        model.add(Dense(dense_output_size, activation='relu'))
        model.add(Dropout(dense_drop_out_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]