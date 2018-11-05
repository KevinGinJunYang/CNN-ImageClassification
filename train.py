#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.
©2018 Created by Yiming Peng and Bing Xue
"""
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import backend as K
from parser import __loader__
from __future__ import division
from keras.optimizers import Adam
from keras import optimizers
from keras.layers import Conv3D, MaxPooling3D, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt
import numpy as np
import keras as k
import tensorflow as tf
import random


SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)
LEARN_RATE = 0.0001
img_size = (300, 300)
adam = optimizers.Adam(lr=LEARN_RATE)



def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(5,5), padding='same', activation='relu' ,input_shape=(300, 300, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Convolution2D(filters=32,kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Convolution2D(filters=64,kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Convolution2D(filters=128,kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Convolution2D(filters=256,kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Convolution2D(filters=512,kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1008, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=3, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
    return model


def preprocess_data(img):
    # All image pre-processing is done here
    return img


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    #data augumentation 
    datagenerator = ImageDataGenerator(rotation_range=360, shear_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.2, rescale=1./255)
    #load data in 
    train_batches = datagenerator.flow_from_directory('./drive/My Drive/Template/data/Train_data', target_size=img_size, classes=['tomato', 'strawberry', 'cherry'], batch_size=32, class_mode='categorical',seed=SEED)
    test_batches = datagenerator.flow_from_directory('./drive/My Drive/Template/data/Test_data', target_size=img_size, classes=['tomato', 'strawberry', 'cherry'], batch_size=32, class_mode='categorical',seed=SEED)
    # Training
    history = model.fit_generator(train_batches, shuffle=True, epochs=110, steps_per_epoch=113, validation_data=test_batches, validation_steps=16)
    # summarize history for accuracy 
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("./drive/My Drive/Template/model/model.h5")
    print("Model Saved Successfully.")

if __name__ == '__main__':
  
    model = construct_model()
    model = train_model(model)
    save_model(model)
    