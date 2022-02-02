# Use this script whenever you need to save the cards to a new directory as a tfds object.

import os
import time
import argparse
import itertools
import sys
from datetime import date
from itertools import cycle
import PIL
import PIL
import pathlib
import scipy.io as Mat
import numpy as np
import matplotlib.pyplot as plt
import random, sklearn
import cv2
import tensorflow as tf

print("Tensorflow version is: ", tf.__version__)

data_dir = pathlib.Path(r'/home/pearlstl/cards/source/as_images') # directory the images are in

# Data parameters
batch_size = 32
img_height = 256 # I have no idea what the images' dimensions are
img_width = 256 # but I found these H and W in another file so I hope that's what they represent

# Area for downloading the image dataset into an object.
card_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    seed=123
)
class_names = card_dataset.class_names

# Saves the images into a tfds object for use in other scripts
tfds_card_path = r'/home/woodsj9/cards/cards_tfds'
tf.data.experimental.save(
    dataset=card_dataset,
    path=tfds_card_path
)

'''Tests that the data will load
load_test = tf.data.experimental.load(
    path = tfds_card_path
)

A visual test with the images to make sure it still works
plt.figure(figsize=(10,10))
for images, labels in load_test.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
'''

'''# Creates a BatchDataset object from a directory of images. This is for the training images.
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123, # Could be worth to wait for shuffling until the datset is going to be used
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# This created BatchDataset  object is for the validation images
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2, # Might want to try setting a training, validation, and separate testing
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names
#print("Class names are :", class_names)

# Plot to show the images properly loaded into the object.
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()


# Preparing the data for the model
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 13

# Creates the model for the dataset
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
])

# Runs the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Tests the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)'''
