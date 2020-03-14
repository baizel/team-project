from classifier.models import BasicTensorFlowModel, AnotherModel
from preprocess.script import PreProcessImages
import numpy as np
from PIL import Image, ImageEnhance
import classifier.util as util

if __name__ == '__main__':
    inPathRoot = "data/GTSRB/Final_Training/Images/"
    outPathRoot = "data/processed/resized/jpg/"

    # Uncomment to resize and split images
    # util.batchResizeAndSplit(inPathRoot, outPathRoot, (80, 20))

    """
    Loading a saved model  and predicting an  image from the test data 
    """

    savedModel = BasicTensorFlowModel().loadSavedModel("AnotherOne")
    # Image path needs to change on specific computers as train and test are random
    toPredict = util.readImageForPrediction("genattack/Adversarial.jpg")
    res = savedModel.predict(toPredict)
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
