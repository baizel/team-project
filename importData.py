#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# In[2]:


data_path = r"C:\Users\Glenn\Desktop\GTSD"
img_size = 32


# In[3]:


def load_data(dataset):
    images = []
    classes = []
    rows = pd.read_csv(dataset)
    rows = rows.sample(frac=1).reset_index(drop=True)
    
    
    for i, row in rows.iterrows():
        img_class = row["ClassId"]
        img_path = row["Path"]
        
        try:
            image = os.path.join(data_path, img_path)
            image = cv2.imread(image)

            image_rs = cv2.resize(image, (img_size, img_size), 3)
            R, G, B = cv2.split(image_rs)
            img_r = cv2.equalizeHist(R)
            img_g = cv2.equalizeHist(G)
            img_b = cv2.equalizeHist(B)
            new_image = cv2.merge((img_r, img_g, img_b))
        
        
        
        
            if i % 500 == 0:
                print(f"loaded: {i}")
            images.append(new_image)
            classes.append(img_class)
            
        except Exception as e:
            print(str(e))
            
        X = np.array(images)
        y = np.array(images)
  
            
    return (X, y)


# In[4]:


epochs = 20
learning_rate = 0.001
batch_size = 64


# In[ ]:


train_data = r"C:\Users\Glenn\Desktop\GTSD\Train.csv"
test_data = r"C:\Users\Glenn\Desktop\GTSD\Test.csv"
(trainX, trainY) = load_data(train_data)
(testX, testY) = load_data(test_data)


# In[ ]:


print("UPDATE: Normalizing data")
trainX = train_X.astype("float32") / 255.0
testX = test_X.astype("float32") / 255.0
print("UPDATE: One-Hot Encoding data")
num_labels = len(np.unique(train_y))
trainY = to_categorical(trainY, num_labels)
testY = to_categorical(testY, num_labels)
class_totals = trainY.sum(axis=0)
class_weight = class_totals.max() / class_totals


# In[ ]:





