import os
import random

from classifier.models import ModelTrainer
from genattack.GenAttack import attack as genAttack
from bruteforce.BruteForce import attack as bruteForceAttack
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


if __name__ == '__main__':
    # resizeAndSplitData()
    model = getModelA()
    files = getAllTestFiles("data/processed/resized/test/")
    for i in range(4):
        file = random.choice(files)
        toPredict = util.readImageForPrediction(file)
        # doGenAttack(toPredict,"0002",model)
        minPert, res = bruteForceAttack(toPredict, model)
        label = file.split("/")[-2]
        saveFileName = "bruteForce_" + str(minPert) + "_" + label + "_" + file.split("/")[-1]
        img = util.arrayToImage(res[0])
        img.save(saveFileName)
        # print(util.getPredictedLabel(util.predictedLabelToMap(model.predict(res))))
