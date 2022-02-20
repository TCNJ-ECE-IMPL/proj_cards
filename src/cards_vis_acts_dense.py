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
from tensorflow.keras.models import save_model, load_model, model_from_json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

print("Tensorflow version is: ", tf.__version__)

data_dir = pathlib.Path(r'/home/pearlstl/cards/source/as_images') # directory the images are in

# Data parameters
batch_size = 32
img_height = 256 # I have no idea what the images' dimensions are
img_width = 256 # but I found these H and W in another file so I hope that's what they represent

full_ds = image_dataset_from_directory(
    data_dir,
    shuffle = False,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

class_names = full_ds.class_names
print("Class names are :", class_names)

num_classes = 13

# load dcnn model
model_name = 'model.json'
json_file = open(model_name, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
print("Model loaded...")

# feed any set of trained weights to model
best_wgt_path = 'weights.h5'
model.load_weights(best_wgt_path)
print("Weights loaded onto model...")

model.summary()

# Runs the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

NUM_DENSE = 128
NUM_SUITS = 4
NUM_VALS  = 13
NUM_ROTS  = 360
ROT_MOD   = 12
ROT_DIV   =  2
ROT_LIM   = round(NUM_ROTS/ROT_DIV)
ROWS_PER_VAL = NUM_SUITS*round(ROT_LIM/ROT_MOD)
NUM_ROWS  = NUM_VALS * ROWS_PER_VAL

the_img_arr = np.zeros((1,img_height,img_width,3))          # create a dummy 4D array as a container, with batch size of 1,
                                                            # to match Keras input layer requirement
the_lbl_arr = np.zeros(1)                                   # create a dummy 1-element numpy array for labels, to match Keras input layer requirement

new_img = np.zeros((NUM_ROWS,NUM_DENSE),dtype=np.uint8) # 4 wide (one per suit, do 4 card values, each 360 rotations)

batch_idx   = 0
suit        = 0
val         = 0
rot         = 0
row_idx     = 0
# Pulls out one batch at a time from the dataset
for batch in full_ds:
    # Pulls out one image at a time from the batch
    for idx in range (len(batch[1].numpy())):

        log_act_flag = (rot < ROT_LIM) and ((rot % ROT_MOD)==0)
        if log_act_flag:
            # batch[0] is a 4D tensor of images, BATCH_SIZE x IMG_HGT x IMG_WID x NUM_COLORS
            the_img=batch[0].numpy()[idx]                       # get one image from the batch
            the_img_arr[0,:,:,:] = the_img                      # put the single image into a container, batch size of 1
            
            the_lbl = batch[1].numpy()[idx]                     # get the label from the batch

            layer_name   = 'dense' 
            layer_target = model.get_layer(name=layer_name)
            activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_target.output)
            # model returns the probability for each outcome class as a value between 0 and 1
            activations = activation_model.predict(the_img_arr)
            activations = activations * 255.0 / 20.0
            activations = np.clip(activations, 0, 255)
            activations = activations.astype(np.uint8)
            sys.stdout.write("val=%d, suit=%d, rot=%d\n" % (val, suit, rot))
            # print(activations)

            new_img[row_idx,:]=activations
            row_idx += 1

        rot += 1
        if (rot == 360):
            rot = 0
            suit += 1
            if (suit == 4):
                suit = 0
                val += 1
    batch_idx += 1

cv2.imwrite("dense_acts.png", new_img)
