import tensorflow as tf

from classifier import util
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.training import Model
import numpy as np

loss_object = tf.keras.losses.CategoricalCrossentropy()

'''
The first step is to create perturbations which will be used to distort the original image
resulting in an adversarial image. As mentioned, for this task, the gradients are taken with respect to the image.
input_image: the target image or named original image
input_label: the label of the target image in the target model
model: the target model which would be attacked
isVerbose: output the process, default is False
'''
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

'''
according the target image to generate perturbations
perturbations can be output
input_image: the target image or named original image
model: the target model which would be attacked
isVerbose: output the process, default is False
useMatplotLib: show perturbations in plt, default is False
'''
def create_perturbations(input_image, model: Model, isVerbose=False, useMatplotLib=False):
    if isVerbose:
        print('Get the input label of the image.')
    # get the input label index of the image
    input_image_index = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(input_image))))
    input_image_probs = model.predict(input_image)
    # convert input into one-hot type data output
    label = tf.one_hot(input_image_index, input_image_probs.shape[-1])
    label = tf.reshape(label, (1, input_image_probs.shape[-1]))
    perturbations = create_adversarial_pattern(input_image, label, model, isVerbose)
    # make resulting perturbations visible
    if useMatplotLib:
        plt.figure
        plt.imshow(perturbations[0])
        plt.show()
    return perturbations

'''
generate adversarial examples by adding small perturbations to the original image
it also trys to find the minimum perturbations by while loop
CAREFUL: the value of epsilon effects accuracy and efficiency, but it is not used as an input to this function
input_image: the target image or named original image
model: the target model which would be attacked
isVerbose: output the process, default is False
useMatplotLib: show adversarial examples in plt, default is False
maxIter: Maximum number of iterations, default 500
'''
def fgsm_attack(input_image, model: Model, isVerbose=False, useMatplotLib=False, maxIter=500):
    epsilon = 0
    iterCount = 0
    initPrediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(input_image))))
    prediction = initPrediction
    perturbations = create_perturbations(input_image, model, isVerbose, useMatplotLib)
    adv_x = input_image
    totalPert = 0
    while prediction == initPrediction and maxIter > iterCount:
        if isVerbose:
            print('Now epsilon is ', epsilon)
            print('Now iterCount is ', iterCount)
        # score calculation for FGSM
        totalPert = np.sum((epsilon * np.abs(perturbations))) + totalPert
        adv_x = input_image + (epsilon * perturbations)
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        if useMatplotLib:
            plt.figure()
            plt.imshow(adv_x[0])
            plt.show()
        prediction = int(util.getPredictedLabel(util.predictedLabelToMap(model.predict(adv_x))))
        iterCount = iterCount + 1
        epsilon = 0.1 * iterCount  # increase the value of epsilon to make the difference obvious
    return totalPert, adv_x, prediction != initPrediction
