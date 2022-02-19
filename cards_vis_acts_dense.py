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
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tf_explain.core.activations import ExtractActivations

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

print("Tensorflow version is: ", tf.__version__)

data_dir = pathlib.Path(r'/home/pearlstl/cards/source/as_images') # directory the images are in

# Data parameters
batch_size = 32
img_height = 256 # I have no idea what the images' dimensions are
img_width = 256 # but I found these H and W in another file so I hope that's what they represent
'''
# Area for downloading the whole image dataset into an object.
card_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    seed=123
)
class_names = card_dataset.class_names
'''
'''
# Saves the images into a tfds object for use in other scripts
tfds_card_path = r'/home/woodsj9/cards/cards_tfds'
tf.data.experimental.save(
    dataset=card_dataset,
    path=tfds_card_path
)

# Tests that the data will load
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

# Creates a BatchDataset object from a directory of images. This is for the training images.
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123, # Could be worth to wait for shuffling until the datset is going to be used
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# This created BatchDataset  object is for the validation images
val_ds = image_dataset_from_directory(
    data_dir,
    shuffle = False,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# for element in val_ds:
#     print(type(element[0].numpy()))
#     print(type(element[1].numpy()))

class_names = train_ds.class_names
print("Class names are :", class_names)

# Plot to show the images properly loaded into the object.
# plt.figure(figsize=(10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(class_names[labels[i]])
#         plt.axis('off')
# plt.show()

# plt.figure(figsize=(10,10))
# for images, labels in val_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(class_names[labels[i]])
#         plt.axis('off')
# plt.show()

# Preparing the data for the model
#AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).cache()
#val_ds = val_ds.prefetch(buffer_size=AUTOTUNE).cache()

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
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu',name="blah"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)
])
model.build((32,256,256, 3))
model.summary()

# Runs the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    validation_data=train_ds,
    epochs=1,
    verbose=2
)

the_img_arr = np.zeros((1,img_height,img_width,3))          # create a dummy 4D array as a container, with batch size of 1,
                                                            # to match Keras input layer requirement
the_lbl_arr = np.zeros(1)                                   # create a dummy 1-element numpy array for labels, to match Keras input layer requirement

batch_idx = 0
# Pulls out one batch at a time from the dataset
for batch in val_ds:
    # Pulls out one image at a time from the batch
    for idx in range (len(batch[1].numpy())):

        # batch[0] is a 4D tensor of images, BATCH_SIZE x IMG_HGT x IMG_WID x NUM_COLORS
        the_img=batch[0].numpy()[idx]                       # get one image from the batch
        the_img_arr[0,:,:,:] = the_img                      # put the single image into a container, batch size of 1
        
        the_lbl = batch[1].numpy()[idx]                     # get the label from the batch
        the_lbl_arr[0] = the_lbl_arr                        # put the label into a container, batch size of 1

        layer_name   = 'dense' 
        layer_target = model.get_layer(name=layer_name)
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_target.output)
        # model returns the probability for each outcome class as a value between 0 and 1
        activations = activation_model.predict(the_img_arr)
        print(activations)
        
#        score = model.evaluate(the_img_arr, the_lbl_arr, verbose=2)     # pass the image into the trained network, and check the score
        prediction = model.predict(the_img_arr)     
        out_lbl = np.argmax(prediction)

        if (the_lbl == out_lbl):
            print("PASS: batch_idx=%d, idx=%d, true_label=%d, out_lbl=%d\n" % (batch_idx, idx, the_lbl, out_lbl))
        else:
            sys.stdout.write("FAIL: batch_idx=%d, idx=%d, true_label=%d, out_lbl=%d\n" % (batch_idx, idx, the_lbl, out_lbl))
            print(prediction)
            # plt.figure(figsize=(10,10))
            # plt.imshow(the_img/255.0)
            # plt.title(class_names[the_lbl])
            # plt.axis('off')
            # plt.show()
            # plt.close()

        # Define the Activation Visualization explainer
        data = (the_img_arr, the_lbl_arr)
        explainer = ExtractActivations()
        grid = explainer.explain(data, model, layers_name='blah')
        dims = grid.shape

        scale_factor = dims[0]/256;
        the_img_scaled = cv2.resize(the_img, None, fx= scale_factor, fy= scale_factor, interpolation= cv2.INTER_LINEAR)

        new_img = np.zeros((dims[0],dims[1]*2,3)) # twice as wide
        rgb_img = np.zeros((dims[0],dims[1]*2,3)) # twice as wide
        new_img[:,0:dims[0],:]= the_img_scaled
        new_img[:,dims[0]:2*dims[1],0]= grid
        new_img[:,dims[0]:2*dims[1],1]= grid
        new_img[:,dims[0]:2*dims[1],2]= grid
        tmp = new_img[:,:,0]
        rgb_img[:,:,0] = new_img[:,:,2]
        rgb_img[:,:,1] = new_img[:,:,1]
        rgb_img[:,:,2] = new_img[:,:,0]

        filename = 'act_%04d_%02d_%s.png' % (batch_idx, idx, class_names[the_lbl])
        explainer.save(rgb_img, '.', filename)
        #print( 'Saved ', filename )

    batch_idx += 1


