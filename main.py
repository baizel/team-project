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


class AttackReturn:
    def __init__(self, attackedImage, minPert, wasSuccessFull):
        self.attackedImage = attackedImage
        self.minPert = minPert
        self.wasSuccessFull = wasSuccessFull


def getAllTestFiles(rootDir):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(rootDir)
             for filename in files
             if filename.endswith(".jpg")]
    return files


def getModelA():
    m = ModelTrainer()
    m.loadSavedModel("ModelA.h5")
    return ModelTrainer().loadSavedModel("ModelA.h5")


def getModelB():
    return ModelTrainer().loadSavedModel("ModelB.h5")


def resizeAndSplitData():
    inPathRoot = "data/GTSRB/Final_Training/Images/"
    outPathRoot = "data/processed/resized/jpg/"

    util.batchResizeAndSplit(inPathRoot, outPathRoot, (80, 20))


def doBruteForce(imgArr, model):
    minPert, res = bruteForceAttack(imgArr, model)
    return AttackReturn(res, minPert, True)


def doGenAttack(imgArr, targetLabel, trainedModel):
    m = genAttack(imgArr, targetLabel, mutationRate=0.3, noiseLevel=0.012, populationSize=20, numberOfGenerations=300, model=trainedModel)
    return AttackReturn(m.image, m.getPerturbation(), m.isAttackSuccess)


def doFSGMAttack(imgArr, model):
    image = __fgsm_preprocess(imgArr[0])
    eps, adv_x, isFound = fgsmAttack(image, model)
    return adv_x, isFound


def __fgsm_preprocess(image):
    image = tf.cast(image, tf.float64)
    image = image[None, ...]
    return image


if __name__ == '__main__':
    # resizeAndSplitData()
    model = getModelA()
    image = util.readImageForPrediction("data/processed/resized/test/00033/00000_00011.jpg")

    files = util.getAllTestFiles("data/processed/resized/test/")
    for i in range(4):
        file = random.choice(files)
        toPredict = util.readImageForPrediction(file)
        # mem = doGenAttack(toPredict,"0002",model)
        res, isFound = doFSGMAttack(toPredict, model)
        print("Did FGSM find an example: ", isFound)
        # minPert, res = bruteForceAttack(toPredict, model)
        label = file.split("/")[-2]
        # saveFileName = "bruteForce_" + str(minPert) + "_" + label + "_" + file.split("/")[-1]
        img = util.arrayToImage(res[0], True)
        # img.save(saveFileName)
        # print(util.getPredictedLabel(util.predictedLabelToMap(model.predict(res))))
