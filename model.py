#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:41:49 2017

@author: changhsin-wen
"""

import csv
import cv2
import numpy as np

lines = []

# Load data

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
  for i in range(3):
    raw_data_path = line[0]
    filename = raw_data_path.split('/')[-1]
    current_path = 'data/IMG/'+filename.split('\\')[-1]
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
 # augment dataset  
 
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)
       
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Use transfer learning implement Nvidia model and add dropout to prefent overfitting save to model.h5

from keras.models import Sequential
from keras.layers import Flatten ,Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import optimizers

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04),loss='mse')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')

    
    
