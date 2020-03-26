import tensorflow as tf

from classifier import util
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.training import Model
import numpy as np

loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label, model: Model, isVerbose=False):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    if isVerbose:
        print('Get the gradients of the loss w.r.t to the input image.')
    gradient = tape.gradient(loss, input_image)
    if isVerbose:
        print('Get the sign of the gradients to create the perturbation')
    signed_grad = tf.sign(gradient)
    return signed_grad


def create_perturbations(input_image, model: Model, isVerbose=False, useMatplotLib=False):
    if isVerbose:
        print('Get the input label of the image.')
    input_image_index = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(input_image))))
    input_image_probs = model.predict(input_image)
    label = tf.one_hot(input_image_index, input_image_probs.shape[-1])
    label = tf.reshape(label, (1, input_image_probs.shape[-1]))
    perturbations = create_adversarial_pattern(input_image, label, model, isVerbose)
    # comment these three lines if u don't want pictures
    if useMatplotLib:
        plt.figure
        plt.imshow(perturbations[0])
        plt.show()
    return perturbations


def fgsm_attack(input_image, model: Model, isVerbose=False, useMatplotLib=False, maxIter=100000):
    epsilon = 0
    iterCount = 0
    initPrediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(input_image))))
    prediction = initPrediction
    perturbations = create_perturbations(input_image, model, isVerbose, useMatplotLib)
    adv_x = input_image
    while prediction == initPrediction and maxIter > iterCount:
        if isVerbose:
            print('Now epsilon is ', epsilon)
            print('Now iterCount is ', iterCount)
        adv_x = input_image + epsilon * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        if useMatplotLib:
            plt.figure()
            plt.imshow(adv_x[0])
            plt.show()
        prediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(adv_x))))
        iterCount = iterCount + 1
        epsilon = 0.1 * iterCount  # change the value(increase 0.01) to make the different obvious

    return epsilon, adv_x, prediction != initPrediction
