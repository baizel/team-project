#!/usr/bin/python3
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import argparse
import random

from classifier.util import AttackReturn
from classifier import util
from genattack.GenAttack import attack as genAttack
from bruteforce.BruteForce import attack as bruteForceAttack
import main

"""
Script to be run from an interactive terminal
"""


def printSummary(attackName, attackReturn: AttackReturn, trainedModel):
    outLabel = util.getPredictedLabel(util.predictedLabelToMap(trainedModel.predict(attackReturn.attackedImage)))
    print("{} Attack Summary".format(attackName))
    print("Was Attack Successful {}".format(attackReturn.wasSuccessFull))
    print("Time Took: {:.2f}s".format(attackReturn.time))
    print("Minimum Perturbation: {:.2f}".format(attackReturn.minPert))
    print("Output Label after Attack: {}".format(outLabel))


def doGenAttack(imgArr, targetLabel, trainedModel, verbose=False):
    ret = main.doGenAttack(imgArr, trainedModel, targetLabel,verbose)
    util.arrayToImage(ret.attackedImage, verbose)
    printSummary("GenAttack", ret, trainedModel)


def doBruteForceAttack(imgArr, trainedModel, verbose=False):
    ret = main.doBruteForce(imgArr, trainedModel)
    util.arrayToImage(ret.attackedImage, verbose)
    printSummary("Brute Force", ret, trainedModel)


def doFGSMAttack(imgArr, trainedModel, verbose=False):
    ret = main.doFSGMAttack(imgArr, trainedModel,verbose)
    util.arrayToImage(ret.attackedImage, verbose)
    printSummary("FGSM", ret, trainedModel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Name of the trained model", dest="model", required=True, metavar="FILE")
    parser.add_argument("-g", "--genattack", help="Perform Gen Attack", dest="gen", action="store_true")
    parser.add_argument("-b", "--bruteforce", help="Perform Brute Force", dest="brute", action="store_true")
    parser.add_argument("-f", "--fgsm", help="Perform FGSM attack", dest="fgsm", action="store_true")
    parser.add_argument("-i", "--image", help="Image for the attack", dest="image", default=None)
    parser.add_argument("-it", "--iteration", help="Number of images to randomly select", dest="iter", default=0, type=int)
    parser.add_argument("-d", "--dir", help="Root Dir of images to be chosen from", dest="dir", default=None)
    parser.add_argument("--targetlabel", help="Target label for Gen attack", dest="targetlabel", choices=util.CLASS_NAMES)
    parser.add_argument("-v", "--verbose", help="Show Image before and after the attack", dest="isVerbose", action="store_true")

    args = parser.parse_args()
    modelName = args.model

    if modelName.split(".")[-1] != "h5":
        raise AssertionError("File needs to have extension h5")
    if [args.gen, args.brute, args.fgsm].count(True) != 1:
        raise AssertionError("Must only choose one attack, (-g,-f,-b)")

    if args.iter != 0:
        if args.dir is None:
            raise AssertionError("Directory for images must be given with iteration param ")
    else:
        if args.image is None:
            raise AssertionError("If iteration is 0 then image param must be given (--image)")
        args.iter = -1  # Single image i chosen by user

    from classifier.models import ModelTrainer  # Slow import so loaded later

    model = ModelTrainer()
    model = model.loadSavedModel(args.model)

    for i in range(max(1, int(args.iter))):
        if args.iter == -1:
            imgPath = args.image
        else:
            imgPath = random.choice(util.getAllTestFiles(args.dir))
        imgArr = util.readImageForPrediction(imgPath)
        if args.gen:
            if args.targetlabel is None:
                raise AssertionError("Target label must be given for this attack")
            print("Attacking Image {} with GenAttack for target label: {}".format(imgPath, args.targetlabel))
            doGenAttack(imgArr, args.targetlabel, model, args.isVerbose)
        elif args.brute:
            print("Attacking Image {} with BruteForce".format(imgPath))
            doBruteForceAttack(imgArr, model, args.isVerbose)
        elif args.fgsm:
            doFGSMAttack(imgArr, model, args.isVerbose)
