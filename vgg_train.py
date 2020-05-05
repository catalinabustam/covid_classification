from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os.path
from os import path

import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

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
y_train_c = to_categorical(y_train, num_classes=3)

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
y_test_c = to_categorical(y_test, num_classes=3)

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
	# load the VGG16 network, ensuring the head FC layer sets are left
	# off
	baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
	# construct the head of the model that will be placed on top of the
	# the base model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(64, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(3, activation="softmax")(headModel)
	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=baseModel.input, outputs=headModel)
	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in baseModel.layers:
		layer.trainable = False
	return model

model= vgg_model()


#opt=tf.keras.optimizers.Adam(lr=learning_rate)

# ********************************************
# Train model
#*********************************************
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train,y_train_c, epochs=epochs, verbose=1, validation_data= (x_test, y_test_c), class_weight=class_weights)

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