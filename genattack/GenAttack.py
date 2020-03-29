from typing import List

import numpy as np
from tensorflow.python.keras.engine.training import Model  # for type hinting

from classifier import util
from classifier.models import ModelTrainer


class Member:
    def __init__(self, image, appliedMask):
        self.image = image
        self.appliedMask = appliedMask
        self.isAttackSuccess = False

    def getPerturbation(self):
        return np.sum(self.appliedMask)


def creatCircularMask(h, w, center=None, radius=None):
    """
    Creates a a true/false array where the true values form a circle from the center position with a given radius
    eg out put of creatCircularMask(5,5,radius=2)
       [
       [F,F,T,F,F],
       [F,T,T,T,F],
       [T,T,T,T,T],
       [F,T,T,T,F],
       [F,F,T,F,F]
       ]

    :param h: height
    :param w: width
    :param center: center position of the circular mask
    :param radius: radius if the circle
    :return: an array of true/false value
    """
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    distFromCenter = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = distFromCenter <= radius
    return mask


def geMutation(x: Member, maxDelta, radius):
    """
    Mutates a given member
    :param x: Image as an array with shape of (x,x,3)
    :param maxDelta: maximum change to any given pixel value
    :param radius: radius of mask that is applied on the image
    :return: a mutated image with more noise added returns a member
    """
    newMember = createAdversarialExample(x.image, maxDelta * -1, maxDelta, radius)
    newMask = x.appliedMask + newMember.appliedMask
    return Member(newMember.image, newMask)


def crossover(p1: Member, p1Score, p2: Member, p2Score):
    """
    Function to make a child image from two images.
    Works by choosing one pixel from either parent depending on their probability
    :param p1:
    :param p1Score:
    :param p2:
    :param p2Score:
    :return:
    """
    probs = p1Score / (p1Score + p2Score)
    p = [probs, 1 - probs]
    pool = [p1, p2]
    retImage = np.zeros((1, 40, 40, 3))
    retMask = np.zeros((40, 40, 3))
    for i in range(3):
        for r in range(40):
            for c in range(40):
                index = np.random.choice([0, 1], p=p)
                feature = pool[index]
                retImage[0][r][c][i] = feature.image[0][r][c][i]
                retMask[r][c][i] = feature.appliedMask[r][c][i]
    return Member(retImage, retMask)


def getFitness(member: Member, model, targetIndex):
    """
    Returns the fitness of the image. Currently it is just the score of
    the target label given by the trained model.
    :param member: The member object that is to be tested
    :param model: The trained model that can be used to predict the given image
    :param targetIndex: The target label for the attack
    :return: float, a score of fitness
    """
    predict = model.predict(member.image)
    # print(predict)
    indx = int(targetIndex)
    score = predict[0][indx]
    total = 0
    for count, val in enumerate(predict[0]):
        if count != indx:
            total = total + val
    return score


def createAdversarialExample(xOriginal, minDelta, maxDelta, radius):
    """
    Adds noise to a given image. The noise is random and applied
    in a circular mask with size of radius. The circle mask center will be the
    center of the image (h/2,w/2) or (20,20) for 40x40 image. The noise is just random
    values added to each pixel value
    :param xOriginal: The image to be edited
    :param minDelta:  min amount of noise
    :param maxDelta: max amount of noise
    :param radius: radius of the circle for the mask
    :return: A member object with modified image and a perturbation level
    """
    reshaped = xOriginal  # xOriginal.reshape((1, 3, 40, 40))
    pert = np.zeros((40, 40, 3))
    for i in range(3):
        values = np.random.uniform(low=minDelta, high=maxDelta, size=(40, 40))
        mask = creatCircularMask(40, 40, radius=radius)
        values[~mask] = 0
        pert[:, :, i] = np.abs(values)
        reshaped[0][:, :, i] = values + reshaped[0][:, :, i]
    return Member(reshaped, pert)


def attack(x, targetLabel: str, noiseLevel, mutationRate, populationSize, numberOfGenerations, model: Model, isVerbose = False) -> Member:
    """
    Gen Attack to attack an image to find an example that misclassifies the original image as the target .
    :param isVerbose: Prints output when running the attack
    :param x: The original Image
    :param targetLabel: The target label to find
    :param noiseLevel: The median noise Level to add to an image
    :param mutationRate: the probability to cause a mutation rate
    :param populationSize: the population of images per generations
    :param numberOfGenerations: the umber of generations to find the example
    :param model: the model to attack
    :return: a Member object, either one that was successful or the one closest to being successful at the end of the generation
    """
    prediction = util.getPredictedLabel(util.predictedLabelToMap(model.predict(x)))
    if prediction == targetLabel:
        raise AssertionError("Given image predicted same as target label")
    population = [None] * populationSize
    for i in range(populationSize):
        population[i] = createAdversarialExample(x, -1 * noiseLevel, noiseLevel, 3)

    for generation in range(numberOfGenerations):
        allFitness = [None] * populationSize
        for j in range(populationSize):
            allFitness[j] = getFitness(population[j], model, targetLabel)
        xAdvMember: Member = population[int(np.argmax(allFitness))]

        prediction = util.getPredictedLabel(util.predictedLabelToMap(model.predict(xAdvMember.image)))
        print("Prediction for Generation {} is {}".format(generation, prediction)) if isVerbose else None
        if prediction == targetLabel:
            xAdvMember.isAttackSuccess = True
            print("Found example in {} generations".format(generation)) if isVerbose else None
            return xAdvMember
        population[0] = xAdvMember
        probs = softMax(allFitness)
        for j in range(1, populationSize):
            allIndex = np.arange(populationSize)
            parent1Index = np.random.choice(allIndex, p=probs)
            parent2Index = np.random.choice(allIndex, p=probs)
            child = crossover(population[parent1Index], allFitness[parent1Index], population[parent2Index], allFitness[parent2Index])
            mutated = child
            if np.random.choice([True, False], p=[mutationRate, 1 - mutationRate]):
                mutated = geMutation(child, noiseLevel, generation % 40)
            population[j] = mutated
    print("No examples found, with GenAttack") if isVerbose else None
    return population[0]


def softMax(x: List[int]):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    toPredict = util.readImageForPrediction("../data/processed/resized/test/00004/00012_00025.jpg")

    model = ModelTrainer().loadSavedModel("../ModelA.h5")
    a = attack(toPredict, "00001", mutationRate=0.8, noiseLevel=0.012, populationSize=20, numberOfGenerations=300, model=model)

    # a = attack(toPredict, "00001", 0.022, 25, 3000, model)

    img = util.arrayToImage(a.image)
    img.show()
    # img.save("Adversarial_4_00012_00025_32_tet.jpg")
    # imgs = []
    # for i in range(43):
    #     img = Image.open("gif/" + str(i) + ".jpg")
    #     imgs.append(img)
    # imgs[0].save("test1.gif", save_all=True, append_images=imgs[1:], optimize=True, duration=500, loop=0)
