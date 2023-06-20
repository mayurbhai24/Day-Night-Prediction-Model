#!/usr/bin/env python
# coding: utf-8

# # Day & Night Prediction Model

# Import Computer Vision and Machine Learning libaries

# In[1]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# The goal of this project is to make a Computer Vision and Machine Learning training model to determine wheather an image was taken in the day or night. At the end, we will use our model and sample images to test our model and determine our accuray score.

# ### 1. Prepare Dataset

# `prepare_dataset(imagesPath)` function takes in the path to the directory which contains two subdirectories `day` and `night`. Each of these subdirectories contain corresponding images, e.g., day directory contains all the day images and night contains night images. The function returns two lists; `image_list` that contains all the images and `labels` which contains either of two values, e.g., `day` or `night`.

# In[5]:


def prepare_dataset(imagesPath):
    labels = []
    images = []
    
    dayimagePath = imagesPath+"/day"
    names = os.listdir(dayimagePath)

    # Day Images
    for name in names:
        labels.append(name.split(".")[0])
        
    for name in names:
        image_name = os.path.join(dayimagePath, name)
        images.append(cv.imread(image_name))
    
    nightimagePath = imagesPath+"/night"
    names = os.listdir(nightimagePath)  
    
    # Night Images
    for name in names:
        labels.append(name.split(".")[0])
        
    for name in names:
        image_name = os.path.join(nightimagePath, name)
        images.append(cv.imread(image_name))
        
    return images, labels


# Image_list contains all the images and labels that contains corresponding label for that image. For example `image_list[0]` contains an array representing the day image and `labels[0]` contains'day' as a string.

# In[6]:


image_list, labels = prepare_dataset("./images/")

print(len(image_list))
print(len(labels))


# ### 2. Resize an Image

# `resize_images(image)` function resizes an image to a size of (600,600) and returns a 1D array of that image.

# In[7]:


def resize_images(image):
    image_resize = []
    size = (600, 600)
    
    for images in image:
        image_resize.append(cv.resize(images, size).flatten())
    
    return image_resize


# ### 3. Encode Labels to Numeric Values

# Machine learning algorithms understand only numeric values, convert labels which are in string into numeric values. `label_images(label)` function takes a label in string and converts it into a numeric value; `day = 1` and `night = 0`

# In[8]:


def label_images(label):
    label_numeric =[]
    
    imagePath = './images/day'
    names = os.listdir(imagePath)
    for i in range(len(names)):
        label_numeric.append(1)
        
    imagePath = './images/night'
    names = os.listdir(imagePath)
    for i in range(len(names)):
        label_numeric.append(0)
        
    return label_numeric


# ### 4. Preprocess Images and Labels 

# Preprocess each image so that each image is a 1D array of (600,600) and labels are numeric values. This function returns two lists; `image_list` containing all the resized images and `label_list` containing all the corresponding numeric labels for these images.

# In[9]:


def preprocess_images(image_list, labels):
    return resize_images(image_list), label_images(labels)
    


# Populate the `preprocessed_images` and `preprocessed_labels`. Now have `preprocessed_images` and `preprocessed_labels` split into train and test datasets.

# In[10]:


preprocessed_images, preprocessed_labels = preprocess_images(image_list, labels)

print(len(preprocessed_images))
print(len(preprocessed_labels))
print(preprocessed_labels)
print(np.shape(preprocessed_images))


# ### 5. Split into Train and Test datasets

# Split data into train and test datasets so that we can use train dataset for training our model and test dataset for testing our model's accuracy.

# In[11]:


train_X_label, test_X_label, train_y_label, test_y_label = train_test_split(preprocessed_images, preprocessed_labels, test_size=0.25)


# ### 6. Training a KNN Model

# `KNeighborsClassifier` model to train the data.

# In[13]:


knn = KNeighborsClassifier()
knn.fit(train_X_label, train_y_label)


# ### 7. Testing the Accuracy

# Test model using test dataset to predict.

# In[15]:


predictions = knn.predict(test_X_label)
score = accuracy_score(predictions, test_y_label)
print(score*100, "%")
print(np.shape(test_X_label))


# ### 8. Final Testing

# Using two sample images, `test_day_image.jpg` and `test_night_image.jpg`, we are testing whether my model predicts them successfully or not.

# In[16]:


# Testing night image
night_image = []
night_label = []

size = (600,600)

image_test = cv.imread('./images/test_night_image.jpg')
night_image.append(cv.resize(image_test, size).flatten())
night_label.append(0)


predictions = knn.predict(night_image)
score = accuracy_score(predictions, night_label)
print(score*100, "%")


# In[17]:


# Testing day image
day_image = []
day_label = []

size = (600,600)

image_test = cv.imread('./images/test_day_image.jpg')
day_image.append(cv.resize(image_test, size).flatten())
day_label.append(1)


predictions = knn.predict(day_image)
score = accuracy_score(predictions, day_label)
print(score*100, "%")


# In[ ]:




