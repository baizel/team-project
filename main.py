from classifier.models import ModelTrainer
from genattack.GenAttack import attack as genAttack
from preprocess.script import PreProcessImages
import numpy as np
from PIL import Image, ImageEnhance
import classifier.util as util


def getModelB():
    return ModelTrainer().loadSavedModel("ModelB.h5")


def getModelA():
    return ModelTrainer().loadSavedModel("ModelA.h5")


def resizeAndSplitData():
    inPathRoot = "data/GTSRB/Final_Training/Images/"
    outPathRoot = "data/processed/resized/jpg/"

    # Uncomment to resize and split images
    util.batchResizeAndSplit(inPathRoot, outPathRoot, (80, 20))


def doGenAttack(imgArr, targetLabel, trainedModel):
    genAttack(imgArr, targetLabel, 0.012, 20, 30, trainedModel)


if __name__ == '__main__':
    """
    Loading a saved model  and predicting an  image from the test data 
    """

    model = getModelA()
    # Image path needs to change on specific computers as train and test are random
    toPredict = util.readImageForPrediction("genattack/Adversarial_32_00003_00009_1.jpg")
    res = model.predict(toPredict)
    print(util.getPredictedLabel(util.predictedLabelToMap(res)))

    """
    Basic perturbation off changing brightness of image to miss classify the image
    """
    # Image path needs to change on specific computers as train and test are random

    # edit = ImageEnhance.Brightness(newImg)
    # edited = edit.enhance(0.00)
    # edited.show()
    # edited.save("Test.jpg")
    # toPredict = util.readImageForPrediction("data/processed/resized/test/00040/00002_00016.jpg")
    # res = savedModel.predict(toPredict)
    # print(util.getPredictedLabel(util.predictedLabelToMap(res)))
