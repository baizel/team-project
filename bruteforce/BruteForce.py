import numpy as np
import time

from classifier import util


def attack(imgArr, model, granularity=100):
    """
    Brute force attack, Gradually adds noise to an image until its misclassified
    :param imgArr: The image as an numpy array
    :param model: The model to attack
    :param granularity: the factor to reduce the noise level as the default noise level is between 0 and 1
    :return:
    """
    assert granularity != 0, "granularity cannot be zero"
    assert imgArr.shape == (1, 40, 40, 3), "Expected shape of image array (1,40,40,3)"
    totalPerturbation = 0
    iterCount = 0
    initPrediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(imgArr))))
    prediction = initPrediction
    while prediction == initPrediction:
        iterCount = iterCount + 1
        for channel in range(3):
            for r in range(40):
                for c in range(40):
                    pixelVal = imgArr[0][r][c][channel]
                    randVal = np.random.random(1)[0] / granularity
                    totalPerturbation = totalPerturbation + randVal
                    pixelVal = pixelVal + (randVal * np.random.choice([1, -1], p=[0.5, 0.5]))
                    imgArr[0][r][c][channel] = pixelVal
        prediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(imgArr))))
    return totalPerturbation, imgArr
