import numpy as np
import time

from classifier import util


def attack(imgArr, model, granularity=100):
    assert granularity != 0, "granularity cannot be zero"
    assert imgArr.shape == (1, 40, 40, 3), "Expected shape of image array (1,40,40,3)"
    totalPerturbation = 0
    iterCount = 0
    initPrediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(imgArr))))
    prediction = initPrediction
    start = time.time()
    while prediction == initPrediction:
        iterCount = iterCount + 1
        for channel in range(3):
            for r in range(40):
                for c in range(40):
                    pixelVal = imgArr[0][r][c][channel]
                    randVal = np.random.random(1)[0] / granularity
                    totalPerturbation = totalPerturbation + randVal
                    pixelVal = pixelVal + randVal
                    imgArr[0][r][c][channel] = pixelVal
        prediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(imgArr))))
    end = time.time()
    print(totalPerturbation, " time = ", end - start,prediction)
    return totalPerturbation,imgArr
