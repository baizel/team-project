import os
import random

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from classifier.models import ModelTrainer
from genattack.GenAttack import attack as genAttack
from bruteforce.BruteForce import attack as bruteForceAttack
from fgsm.FGSM import fgsm_attack as fgsmAttack
import classifier.util as util

def getAllTestFiles(rootDir):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(rootDir)
             for filename in files
             if filename.endswith(".jpg")]
    return files


def getModelA():
    return ModelTrainer().loadSavedModel("ModelA.h5")


def getModelB():
    return ModelTrainer().loadSavedModel("ModelB.h5")


def resizeAndSplitData():
    inPathRoot = "data/GTSRB/Final_Training/Images/"
    outPathRoot = "data/processed/resized/jpg/"

    util.batchResizeAndSplit(inPathRoot, outPathRoot, (80, 20))


def doGenAttack(imgArr, targetLabel, trainedModel):
    genAttack(imgArr, targetLabel, 0.012, 20, 300, trainedModel)

def fgsm_preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (40, 40))
    image = image[None, ...]
    return image

if __name__ == '__main__':
    #resizeAndSplitData()
    pretrained_model = getModelA()
    #print(type(pretrained_model))

    mpl.rcParams['figure.figsize'] = (8, 8)
    mpl.rcParams['axes.grid'] = False
    #choose a speficied image
    #image0 = mpimg.imread('data/processed/resized/jpgtest/00000/00000_00001.jpg')
    image_raw = tf.io.read_file('data/processed/resized/jpgtest/00000/00000_00001.jpg')
    image = tf.image.decode_image(image_raw)
    image = fgsm_preprocess(image)
    image_probs = pretrained_model.predict(image)

    #image = util.readImageForPrediction('data/processed/resized/jpgtest/00000/00000_00001.jpg')
    #image_probs = pretrained_model.predict(image)
    #image2 = tf.image.decode_image(image)
    #image2_probs = pretrained_model.predict(image2)
    #print(type(image_raw), image_raw.shape)
    print(type(image),image.shape)
    plt.figure()
    plt.imshow(image[0])
    plt.show()
    eps, adv_x = fgsmAttack(image, pretrained_model)

    #resizeAndSplitData()
    #model = getModelA()
    #print(type(model))
    #files = getAllTestFiles("data/processed/resized/jpgtest/")
    #for i in range(4):
        #file = random.choice(files)
        #toPredict = util.readImageForPrediction(file)
        #doGenAttack(toPredict,"0002",model)
        #minPert, res = bruteForceAttack(toPredict, model)
        #label = file.split("/")[-2]
        #saveFileName = "bruteForce_" + str(minPert) + "_" + label + "_" + file.split("/")[-1]
        #img = util.arrayToImage(res[0])
        #img.save(saveFileName)
        #print(util.getPredictedLabel(util.predictedLabelToMap(model.predict(res))))
