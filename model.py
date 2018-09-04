import csv
import cv2
import numpy as np 
import sklearn
import os
from sklearn.utils import shuffle
from random import randint

#Prepare training and validation sets
samples = []
with open('recording/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #randomly removing 40% of the lines with small (less than 0.8) steering angles
        if (abs(float(line[3])) > .8 or randint(0,99)>40):
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)
shuffle(train_samples)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.1 #steering correction
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread('recording/IMG/'+batch_sample[0].split('/')[-1])
                left_image = cv2.imread('recording/IMG/'+batch_sample[1].split('/')[-1])
                right_image = cv2.imread('recording/IMG/'+batch_sample[2].split('/')[-1])
                #Images pre-processing consists of converting to YUV color space
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                steering_center = float(batch_sample[3])
                #Apply correction factor to the steering angle associated with side cameras images
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                images.extend([center_image,left_image,right_image])
                angles.extend([steering_center,steering_left,steering_right])    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and train the model using the generator function
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=3)

model.save('model.h5')
