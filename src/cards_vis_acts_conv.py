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
from tf_explain.core.activations import ExtractActivations

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

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

the_img_arr = np.zeros((1,img_height,img_width,3))          # create a dummy 4D array as a container, with batch size of 1,
                                                            # to match Keras input layer requirement
the_lbl_arr = np.zeros(1)                                   # create a dummy 1-element numpy array for labels, to match Keras input layer requirement

batch_idx = 0
# Pulls out one batch at a time from the dataset
for batch in full_ds:
    # Pulls out one image at a time from the batch
    for idx in range (len(batch[1].numpy())):

        # batch[0] is a 4D tensor of images, BATCH_SIZE x IMG_HGT x IMG_WID x NUM_COLORS
        the_img=batch[0].numpy()[idx]                       # get one image from the batch
        the_img_arr[0,:,:,:] = the_img                      # put the single image into a container, batch size of 1
        
        the_lbl = batch[1].numpy()[idx]                     # get the label from the batch
        the_lbl_arr[0] = the_lbl_arr                        # put the label into a container, batch size of 1

        prediction = model.predict(the_img_arr)     
        out_lbl = np.argmax(prediction)

        if (the_lbl == out_lbl):
            sys.stdout.write("PASS: batch_idx=%d, idx=%d, true_label=%d, out_lbl=%d\n" % (batch_idx, idx, the_lbl, out_lbl))
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


