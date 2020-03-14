from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.models import load_model
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from classifier import util as datasetUtil

IMG_HEIGHT, IMG_WIDTH = (40, 40)
batch_size = 128
epochs = 3


class AnotherModel:
    def __init__(self):
        model = Sequential()
        # add the convolutional layer
        # filters, size of filters,input_shape,activation_function
        model.add(Conv2D(60, (5, 5), input_shape=(40, 40, 3), activation='relu'))
        model.add(Conv2D(60, (5, 5), input_shape=(40, 40, 3), activation='relu'))
        # pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # add another convolutional layer
        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(Conv2D(30, (3, 3), activation='relu'))
        # pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Dropout(0.5))

        # Flatten the image to 1 dimensional array
        model.add(Flatten())
        # add a dense layer : amount of nodes, activation
        model.add(Dense(500, activation='relu'))
        # place a dropout layer
        # 0.5 drop out rate is recommended, half input nodes will be dropped at each update
        model.add(Dropout(0.5))
        # defining the ouput layer of our network
        model.add(Dense(43, activation='softmax'))
        self.__model = model
        self.__model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.__model.summary()

    def train(self,
              trainTestSplit=(80, 20),
              runPreProcessor=False,
              rawImagePath=os.path.join("data", "GTSRB", "Final_Training", "Images"),
              processedOutPath=os.path.join("data", "processed", "resized", "jpg")
              ):
        trainDataSet, testDataSet = datasetUtil.getDataSet(rawImagePath, processedOutPath, runPreProcessor, trainTestSplit=trainTestSplit)
        trainDataSet = trainDataSet.batch(batch_size)
        testDataSet = testDataSet.batch(batch_size)
        trainDataSet.shuffle(10000)
        # trainTotal = 31410
        # testTotal = 7799
        history = self.__model.fit(
            trainDataSet.take(245),
            # steps_per_epoch=trainTotal // batch_size,
            epochs=epochs,
            validation_data=testDataSet.take(60),
            # validation_steps=testTotal // batch_size
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        #
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

        return acc, val_acc

    def saveModel(self, fileName="AnotherModel.h5"):
        self.__model.save(fileName)

    def loadSavedModel(self, fileName):
        self.__model = load_model(fileName)
        return self.__model

    def getModel(self):
        return self.__model


class BasicTensorFlowModel:

    def __init__(self):
        self.__model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(),
            Dropout(0.6),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.5),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.6),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(43)
        ])
        self.__model.compile(optimizer='adam',
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                             metrics=['accuracy'])

    def summary(self):
        self.__model.summary()

    def train(self,
              trainTestSplit=(80, 20),
              runPreProcessor=False,
              rawImagePath=os.path.join("data", "GTSRB", "Final_Training", "Images"),
              processedOutPath=os.path.join("data", "processed", "resized", "jpg")
              ):
        trainDataSet, testDataSet = datasetUtil.getDataSet(rawImagePath, processedOutPath, runPreProcessor, trainTestSplit=trainTestSplit)
        trainDataSet = trainDataSet.batch(batch_size)
        testDataSet = testDataSet.batch(batch_size)
        trainDataSet.shuffle(10000)
        # trainTotal = 31410
        # testTotal = 7799
        history = self.__model.fit(
            trainDataSet.take(245),
            # steps_per_epoch=trainTotal // batch_size,
            epochs=epochs,
            validation_data=testDataSet.take(60),
            # validation_steps=testTotal // batch_size
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        #
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

        return acc, val_acc

    def saveModel(self, fileName="basic"):
        self.__model.save(fileName + ".h5")

    def loadSavedModel(self, fileName="basic"):
        self.__model = load_model(fileName + '.h5')
        return self.__model

    def getModel(self):
        return self.__model
