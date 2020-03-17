from typing import List

import numpy as np
from tensorflow.python.keras.engine.training import Model  # for type hinting

from classifier import util
from classifier.models import ModelTrainer


def creatCircularMask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    distFromCenter = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = distFromCenter <= radius
    return mask


def geMutation(x, maxDelta, radius):
    return createAdversarialExample(x, maxDelta * -1, maxDelta, radius)


def crossover(p1, p1Score, p2, p2Score):
    probs = p1Score / (p1Score + p2Score)
    p = [probs, 1 - probs]
    pool = [p1, p2]
    ret = np.zeros((1, 40, 40, 3))
    for i in range(3):
        for r in range(40):
            for c in range(40):
                index = np.random.choice([0, 1], p=p)
                feature = pool[index]
                ret[0][r][c][i] = feature[0][r][c][i]
    return ret


def getFitness(xAdv, model, targetIndex):
    predict = model.predict(xAdv)
    # print(predict)
    indx = int(targetIndex)
    score = predict[0][indx]
    total = 0
    for count, val in enumerate(predict[0]):
        if count != indx:
            total = total + val
    return score


def createAdversarialExample(xOriginal, minDelta, maxDelta, radius):
    reshaped = xOriginal  # xOriginal.reshape((1, 3, 40, 40))
    for i in range(3):
        values = np.random.uniform(low=minDelta, high=maxDelta, size=(40, 40))
        mask = creatCircularMask(40, 40, radius=radius)
        values[~mask] = 0
        reshaped[0][:, :, i] = values + reshaped[0][:, :, i]
    return reshaped


def attack(x, targetLabel: str, mutationRate, populationSize, numberOfGenerations, model: Model):
    population = [None] * populationSize
    for i in range(populationSize):
        population[i] = createAdversarialExample(x, -1 * mutationRate, mutationRate, 3)

    for generation in range(numberOfGenerations):
        allFitness = [None] * populationSize
        for j in range(populationSize):
            allFitness[j] = getFitness(population[j], model, targetLabel)
        xAdv = population[int(np.argmax(allFitness))]
        # img = util.arrayToImage(xAdv[0])

        prediction = util.getPredictedLabel(util.predictedLabelToMap(model.predict(xAdv)))
        print("Prediction for Generation {} is {}".format(generation, prediction))
        if prediction == targetLabel:
            print("Found example in {} generations".format(generation))
            return xAdv
        population[0] = xAdv
        probs = softMax(allFitness)
        for j in range(1, populationSize):
            allIndex = np.arange(populationSize)
            parent1Index = np.random.choice(allIndex, p=probs)
            parent2Index = np.random.choice(allIndex, p=probs)
            child = crossover(population[parent1Index], allFitness[parent1Index], population[parent2Index], allFitness[parent2Index])
            population[j] = geMutation(child, mutationRate, generation % 40)
    print("No examples found, increase generation size")
    return population[0]


def softMax(x: List[int]):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    toPredict = util.readImageForPrediction("../data/processed/resized/test/00004/00012_00025.jpg")

    model = ModelTrainer().loadSavedModel("../ModelA.h5")
    a = attack(toPredict, "00001", 0.022, 25, 3000, model)

    img = util.arrayToImage(a[0])
    img.show()
    img.save("Adversarial_4_00012_00025_32_tet.jpg")
    # imgs = []
    # for i in range(43):
    #     img = Image.open("gif/" + str(i) + ".jpg")
    #     imgs.append(img)
    # imgs[0].save("test1.gif", save_all=True, append_images=imgs[1:], optimize=True, duration=500, loop=0)
