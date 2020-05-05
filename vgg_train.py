import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow.keras as keras
from keras import models
from keras import optimizers
from sklearn.metrics import confusion_matrix
import os.path
from os import path
import csv
import argparse

from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, add, Flatten, Dense
from keras import layers
from keras.applications import VGG16
from keras.layers.core import Flatten, Dense, Dropout, Lambda

# Define data location 
data_path = './data/'
train_csv = data_path + 'train_split_v3.txt'
test_csv = data_path + 'test_split_v3.txt'

# Set classification options
mapping={'normal': 0,'pneumonia': 1,'COVID-19': 2}

# Load train files
#********************************
with open(train_csv) as f:
    trainfiles = f.readlines()

y_train = []
x_train = []
for i in range(len(trainfiles)):
  line = trainfiles[i].split()
  path_f = os.path.join(data_path, 'train', line[1])

 # Some images have errors on dataset generation
  if path.exists(path_f):
    x = cv2.imread(path_f)
    h, w, c = x.shape
    x = x[int(h/6):, :]
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0
    y_train.append(mapping[line[2]])
    x_train.append(x)
y_train = np.array(y_train)
y_train_c = keras.utils.to_categorical(y_train, num_classes=3)

x_train = np.array(x_train)
#*******************************************

# Load test files
#********************************
with open(test_csv) as f:
    testfiles = f.readlines()

y_test = []
x_test = []
for i in range(len(testfiles)):
  line = testfiles[i].split()
  path_f = os.path.join(data_path, 'test', line[1])

 # Some images have errors on dataset generation
  if path.exists(path_f):
    x = cv2.imread(path_f)
    h, w, c = x.shape
    x = x[int(h/6):, :]
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0
    y_test.append(mapping[line[2]])
    x_test.append(x)
y_test = np.array(y_test)
y_test_c = keras.utils.to_categorical(y_test, num_classes=3)

x_test = np.array(x_test)
#*******************************************

# Define traing params

#*******************************************
learning_rate = 0.00002
batch_size =6
display_step = 1
epochs = 25
class_weights=[1, 1, 12]

def vgg_model():
	conv_base = VGG16(weights='imagenet',
         include_top=False,
         input_shape=(224, 224, 3))
	conv_base.trainable = False
	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(3, activation='sigmoid'))
	return model

model= vgg_model()


opt=tf.keras.optimizers.Adam(lr=learning_rate)

# ********************************************
# Train model
#*********************************************
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train,y_train_c, epochs=epochs, verbose=1)

#********************************************
# Save results
#*******************************************
model.save('model_vgg.h5')

pred = model.predict(x_test)

matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))
matrix = matrix.astype('float')
 
print(matrix)

class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                             class_acc[1],
                                                                             class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],

                                                                            ppvs[1],
                                                                             ppvs[2]))