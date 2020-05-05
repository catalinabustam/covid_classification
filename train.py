import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, add, Flatten, Dense
from keras import models
from keras import layers
from keras.applications import VGG16
from keras import optimizers
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from sklearn.metrics import confusion_matrix
import os.path
from os import path
import csv
import argparse
from data import BalanceCovidDataset
from keras_model import keras_model_build
from keras.backend.tensorflow_backend import set_session
                                    # (nothing gets printed in Jupyter, only if you run it standalone)


from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


data_path = './data/'
train_csv = data_path + 'train_split_v3.txt'
test_csv = data_path + 'test_split_v3.txt'

mapping={'normal': 0,'pneumonia': 1,'COVID-19': 2}

with open(test_csv) as f:
    testfiles = f.readlines()

y_test = []
x_test = []
for i in range(len(testfiles)):
    line = testfiles[i].split()
    path_f = os.path.join(data_path, 'test', line[1])
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

print("x_test_size")
print(x_test.shape)
learning_rate = 0.00002
batch_size =6
display_step = 1
epochs = 25
class_weights=[1., 1., 12]

def weighted_categorical_crossentropy(weights):
	def loss(y_true, y_pred):
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)*weights
		return loss
	return loss

generator = BalanceCovidDataset(data_dir=data_path,
                                csv_file=train_csv,
                                covid_percent=0.3,
                                class_weights=class_weights,
                                batch_size= batch_size)

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


total_batch= len(generator)
model = keras_model_build()
#model= vgg_model()
opt=keras.optimizers.Adam(lr=learning_rate)

epochs = 25

opt=tf.keras.optimizers.Adam(lr=0.00001)

weights_test = np.take(class_weights, y_test.astype('int64'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

hist = model.fit(generator, steps_per_epoch=total_batch, epochs=epochs, verbose=1, validation_data= (x_test, y_test_c), class_weight=[1,1,12])

model.save('model.h5')

hist_loss = hist.history["loss"]
hist_accuracy = hist.history["acc"]
hist_accuracy_test = hist.history["val_acc"]
hist_loss_test = hist.history["val_loss"]

np.savetxt("loss_his.csv", hist_loss, delimiter=",", fmt='%s')
np.savetxt("hist_accuracy.csv", hist_accuracy, delimiter=",", fmt='%s')
np.savetxt("loss_test.csv", hist_loss_test, delimiter=",", fmt='%s')
np.savetxt("hist_accuracy_test.csv", hist_accuracy_test, delimiter=",", fmt='%s')

pred = model.predict(x_test)

np.savetxt("pred_test.csv", np.argmax(pred, axis=1), delimiter=",", fmt='%s')
np.savetxt("true_test.csv", y_test, delimiter=",", fmt='%s')
matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))
matrix = matrix.astype('float')
  #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
print(matrix)
  #class_acc = np.array(cm_norm.diagonal())
class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                             class_acc[1],
                                                                             class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],

                                                                            ppvs[1],
                                                                             ppvs[2]))