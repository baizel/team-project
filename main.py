import os
import random
import sys
import time

import tensorflow as tf

from classifier.models import ModelTrainer
from genattack.GenAttack import attack as genAttack
from bruteforce.BruteForce import attack as bruteForceAttack
from fgsm.FGSM import fgsm_attack as fgsmAttack
import classifier.util as util


def timeFunction(attack):
    def timed(*args, **kwargs):
        ts = time.time()
        attackRet: AttackReturn = attack(*args, **kwargs)
        te = time.time()
        attackRet.time = te - ts
        return attackRet

    return timed


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


@timeFunction
def doBruteForce(imgArr, model):
    minPert, res = bruteForceAttack(imgArr, model)
    return AttackReturn(res, minPert, True)


@timeFunction
def doGenAttack(imgArr, model, targetLabel):
    m = genAttack(imgArr, targetLabel, mutationRate=0.3, noiseLevel=0.12, populationSize=25, numberOfGenerations=300, model=model)
    return AttackReturn(m.image, m.getPerturbation(), m.isAttackSuccess)


@timeFunction
def doFSGMAttack(imgArr, model):
    image = __fgsm_preprocess(imgArr[0])
    totPert, adv_x, isFound = fgsmAttack(image, model)
    return AttackReturn(adv_x, totPert, isFound)


def __fgsm_preprocess(image):
    image = tf.cast(image, tf.float64)
    image = image[None, ...]
    return image


if __name__ == '__main__':
    # resizeAndSplitData()
    model = getModelA()
    image = util.readImageForPrediction("data/processed/resized/test/00033/00000_00011.jpg")

    files = util.getAllTestFiles("data/processed/resized/test/")
    attacks = {
        "FGSM": {"functionName": "doFSGMAttack", "args": []},
        "Brute Force": {"functionName": "doBruteForce", "args": []},
        "GenAttack": {"functionName": "doGenAttack", "args": ["00002"]},
    }

    for i in range(4):
        file = random.choice(files)
        toPredict = util.readImageForPrediction(file)
        for key in attacks.keys():
            label = file.split("/")[-2]
            attackFunc = getattr(sys.modules[__name__], attacks[key]["functionName"])
            optArgs = attacks[key]["args"]
            args = (toPredict, model)
            if len(optArgs) != 0:
                if int(optArgs[0]) == int(label):
                    optArgs[0] = "00003"  # Dirty fix to avoid 'Given image predicted same as target label error'
                args = args + tuple(optArgs)
            ret: AttackReturn = attackFunc(*args)
            predictedLabel = util.getPredictedLabel(util.predictedLabelToMap(model.predict(ret.attackedImage)))
            print("Attack: {}, Time taken: {:.3f}(s) Min Perturbation: {:.3f}, Was Attack Successful: {}, Final Prediction: {}".format
                  (key, ret.time, ret.minPert, ret.wasSuccessFull, predictedLabel))
            img = util.arrayToImage(ret.attackedImage)
            originalFileName = file.split("/")[-1]
            # Format : Attack_NAme, perturbed amount, original label, attacked_label, time took to attack, if attack was success, original name
            saveFileName = "{0}_{1:.2f}_{2}_{3}_{4}_{5:.2f}_{6}".format(key.replace(" ", ""), ret.minPert, label, predictedLabel, ret.wasSuccessFull, ret.time, originalFileName)
            path = os.path.join("attack_examples", saveFileName)
            img.save(path)
