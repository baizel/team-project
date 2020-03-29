from __future__ import absolute_import, division, print_function, unicode_literals

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import operator

from PIL import Image

from preprocess.script import PreProcessImages

AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASS_NAMES = np.array(["{:05d}".format(x) for x in range(0, 43)])


class AttackReturn:
    def __init__(self, attackedImage, minPert, wasSuccessFull):
        self.attackedImage = attackedImage
        self.minPert = minPert
        self.wasSuccessFull = wasSuccessFull
        self.time = None


def getAllTestFiles(rootDir):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(rootDir)
             for filename in files
             if filename.endswith(".jpg")]
    return files


def arrayToImage(arr, showImage=False):
    if len(arr.shape) == 4:
        arr = arr[0]
    img = Image.fromarray(np.uint8(arr * 255))
    if showImage:
        img.show()
    return img


def getPredictedLabel(mappedValues):
    key = max(mappedValues.items(), key=operator.itemgetter(1))[0]
    return key


def predictedLabelToMap(predictedLabel):
    mappedLabels = {}
    for i in CLASS_NAMES:
        mappedLabels[i] = predictedLabel[0][int(i)]

    return mappedLabels


def readImageForPrediction(filePath):
    img = tf.io.read_file(filePath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.convert_image_dtype(img, tf.float64)
    return np.asarray(img.numpy()).reshape((1, 40, 40, 3))


def __goBackOneDir(path):
    splitOut = path.split(os.sep)
    splitOut.remove(splitOut[-1])
    out = splitOut[0]
    for x in range(1, len(splitOut)):
        out = os.path.join(out, splitOut[x])
    return out


def batchResizeAndSplit(inPathRoot: str, outPathRoot: str, trainTestSplit=(80, 20)):
    out = __goBackOneDir(outPathRoot)
    preprocessor = PreProcessImages(inPathRoot)
    preprocessor.batchResize(keepAspectRatio=False, outputTargetSize=(40, 40), outputDirRoot=outPathRoot, outFormat="jpg")
    preprocessor.splitDataIntoTrainAndTest(outPathRoot, out, trainTestSplit)


def getDataSet(inPathRoot: str, outPathRoot: str, runPreProcessor=True, trainTestSplit=(80, 20)) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    out = __goBackOneDir(outPathRoot)
    if runPreProcessor:
        batchResizeAndSplit(inPathRoot, outPathRoot, trainTestSplit)
    trainDir = os.path.join(out, "train", "*", "*.jpg")
    testDir = os.path.join(out, "test", "*", "*.jpg")
    train = tf.data.Dataset.list_files(trainDir)
    test = tf.data.Dataset.list_files(testDir)
    return train.map(__processPath, num_parallel_calls=AUTOTUNE), test.map(__processPath, num_parallel_calls=AUTOTUNE)


def __getLabel(filePath):
    return tf.strings.split(filePath, os.sep)[-2] == CLASS_NAMES


def __decodeImg(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.convert_image_dtype(img, tf.float64)
    return img


def __processPath(filePath):
    label = __getLabel(filePath)
    img = tf.io.read_file(filePath)
    img = __decodeImg(img)
    return img, label
