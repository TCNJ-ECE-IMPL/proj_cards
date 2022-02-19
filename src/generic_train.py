import os
import time
import argparse
import itertools
from datetime import date
from itertools import cycle

import scipy.io as Mat
import numpy as np
import matplotlib.pyplot as plt
import random, sklearn
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Activation, multiply, Lambda, BatchNormalization
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam, Nadam, Adagrad, Adadelta, RMSprop
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

#os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Optimizer parameters
OPTIMIZER       = "NADAM"
OPTIMIZER_LR    = 0.0001
OPTIMIZER_BETA1 = 0.9
OPTIMIZER_BETA2 = 0.999
INIACC_VAL      = 0.1 #Used in Adagrad
RHO_VAL         = 0.9 #Used in Adadelta and RMS

# Network architecture
CONV1_FILTERS   = 96
CONV1_KERNEL_SZ = 11
POOL1_POOL_SZ   = 2

CONV2_FILTERS   = 256
CONV2_KERNEL_SZ = 5
POOL2_POOL_SZ   = 2

CONV3_FILTERS   = 384
CONV3_KERNEL_SZ = 3
POOL3_POOL_SZ   = 1

CONV4_FILTERS   = 384
CONV4_KERNEL_SZ = 3
POOL4_POOL_SZ   = 1

CONV5_FILTERS   = 256
CONV5_KERNEL_SZ = 3
POOL5_POOL_SZ   = 2

DENSE1_UNITS    = 4096
DENSE2_UNITS    = 4096

# Model-specific constants
BATCH_SIZE      = 16
EPOCHS          = 100
NUM_CLASSES     = 13
NUM_ANGLES      = 20

# Create initialized 4-dimensional array
X_train = np.zeros((NUM_ANGLES*4*NUM_CLASSES, 256, 256, 3),dtype=np.uint8)
Y_train = np.zeros((NUM_ANGLES*4*NUM_CLASSES, NUM_CLASSES), dtype=np.float)

in_shape = X_train.shape[1:]

# Initialize DCNN as Sequential model
model = Sequential()

#1 - number of conv layers
model.add(BatchNormalization(input_shape=(in_shape)))
model.add(Conv2D(filters=CONV1_FILTERS, kernel_size=CONV1_KERNEL_SZ,
                    activation='relu',
                    padding='valid',
                    strides=(4,4)                    
                    ))
model.add(MaxPooling2D(pool_size=POOL1_POOL_SZ))

#2 - number of conv layers
model.add(BatchNormalization())
model.add(Conv2D(filters=CONV2_FILTERS, kernel_size=CONV2_KERNEL_SZ,
                activation='relu',
                padding='same',
                strides=(1,1)
                ))
model.add(MaxPooling2D(pool_size=POOL2_POOL_SZ))

#3
model.add(BatchNormalization())
model.add(Conv2D(filters=CONV3_FILTERS, kernel_size=CONV3_KERNEL_SZ,
                activation='relu',
                padding='same',
                strides=(1,1)
                ))

#4
model.add(BatchNormalization())
model.add(Conv2D(filters=CONV4_FILTERS, kernel_size=CONV4_KERNEL_SZ,
                activation='relu',
                padding='same',
                strides=(1,1)
                ))

#5
model.add(BatchNormalization())
model.add(Conv2D(filters=CONV5_FILTERS, kernel_size=CONV5_KERNEL_SZ,
                activation='relu',
                padding='valid',
                strides=(1,1)
                ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL5_POOL_SZ))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(units=DENSE1_UNITS, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(units=DENSE2_UNITS, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(units=NUM_CLASSES, activation='softmax'))

print("Input Signal Size: {}".format(in_shape))
print(NUM_CLASSES)
model.summary()

# e.g. in ~/cards/source/as_images/cardval_6
# card_d_6_179_deg.png

idx = 0     # index into data array
# load data into variables

for dummy in range (1):
    for i in range(NUM_ANGLES*NUM_CLASSES):
        for suit in ('s', 'h', 'd', 'c'):
            val = random.randint(1, 13)
            deg = random.randint(0,359)
# for deg in range(NUM_ANGLES):
#     print(deg)
#     for suit in ('s', 'h', 'd', 'c'):
#         for val in range(1,14):
            if val >= 2 and val <= 10:
                val_char = "%d" % val
            elif val == 1:
                val_char = "a"
            elif val == 11:
                val_char = "j"
            elif val == 12:
                val_char = "q"
            elif val == 13:
                val_char = "k"
            filepath = "/home/pearlstl/cards/source/as_images/cardval_%s/card_%s_%s_%03d_deg.png" % (val_char, suit, val_char, deg)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #print( img.dtype) 
            X_train[idx,:,:,:] = img
            Y_train[idx][val-1] = 1.0
            idx = idx + 1
#print( "%d images read" % idx )

# compile model
NAdamX = Nadam(lr=OPTIMIZER_LR, beta_1=OPTIMIZER_BETA1, beta_2=OPTIMIZER_BETA2, epsilon=1e-07, name="Nadam")
model.compile(loss='categorical_crossentropy',
                optimizer=NAdamX,
                metrics=['accuracy'])

# train model
wgt_name = 'wgt_epoch_' + str(EPOCHS) + '.h5'
best_wgt_path = wgt_name
checkpoint_callback = ModelCheckpoint(
    filepath=best_wgt_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

print(X_train.shape)

# train model and save history to a variable
model_hist = model.fit(X_train, Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[checkpoint_callback],
            validation_split=0.2)

# load best weights onto model
model.load_weights(best_wgt_path)

# test/evaluate model
score = model.evaluate(X_train, Y_train, verbose=1)
print("Test Loss: {}".format(score[0]))
print("Test Accuracy: {}".format(score[1]))

