from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import os
from tensorflow_core.python.keras.models import load_model
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from classifier import util as datasetUtil

IMG_HEIGHT, IMG_WIDTH = (40, 40)
batch_size = 128
epochs = 3

SeqModelA = Sequential()
# add the convolutional layer
# filters, size of filters,input_shape,activation_function
SeqModelA.add(Conv2D(60, (5, 5), input_shape=(40, 40, 3), activation='relu'))
SeqModelA.add(Conv2D(60, (5, 5), input_shape=(40, 40, 3), activation='relu'))
# pooling layer
SeqModelA.add(MaxPooling2D(pool_size=(2, 2)))
# add another convolutional layer
SeqModelA.add(Conv2D(30, (3, 3), activation='relu'))
SeqModelA.add(Conv2D(30, (3, 3), activation='relu'))
# pooling layer
SeqModelA.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the image to 1 dimensional array
SeqModelA.add(Flatten())
# add a dense layer : amount of nodes, activation
SeqModelA.add(Dense(500, activation='relu'))
# place a dropout layer
# 0.5 drop out rate is recommended, half input nodes will be dropped at each update
SeqModelA.add(Dropout(0.5))
# defining the ouput layer of our network
SeqModelA.add(Dense(43, activation='softmax'))
SeqModelA.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

SqeModelB = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.4),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.4),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(43)
])
SqeModelB.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


class ModelTrainer:
    def __init__(self, compliedModel=None):
        self.__model = compliedModel

    def summary(self):
        assert  self.__model is not None, "No model loaded or trained"
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

    def saveModel(self, fileName):
        ext = fileName.split(".")[-1]
        assert ext == "h5", "file name must extension type of h5"
        self.__model.save(fileName)

    def loadSavedModel(self, fileName):
        ext = fileName.split(".")[-1]
        assert ext == "h5", "file name must extension type of h5"
        self.__model = load_model(fileName)
        return self.__model

    def getModel(self):
        assert  self.__model is not None, "No model loaded or trained"
        return self.__model
