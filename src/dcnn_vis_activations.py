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
import math
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import save_model, load_model, model_from_json


parser=argparse.ArgumentParser(description='Visualizes using activations')
parser.add_argument('--model', metavar='model',type=str, required=True)
parser.add_argument('--wgts_file',metavar='wgts_file',type=str,required=True)
parser.add_argument('--data_dir',metavar='data_dir',type=str,required=True)
parser.add_argument('--data_path',metavar='data_path',type=str,required=True)


args=parser.parse_args()
model=args.model
wgts_file=args.wgts_file
data_dir=args.data_dir
data_path=args.data_path

home_dir=os.getenv('HOME')+'/'

os.mkdir(data_path)
os.chdir(data_path)



#data parameters
batch_size = 32
img_height = 256 
img_width = 256 
full_ds = image_dataset_from_directory(
    data_dir,
    shuffle = False,
    image_size = (img_height, img_width),
    batch_size = batch_size
)




Decrement=2


#dcnn model
model_name = model
json_file = open(model_name, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
print("Model loaded...")

# feed any set of trained weights to model
best_wgt_path = wgts_file
model.load_weights(best_wgt_path)
print("Weights loaded onto model...")

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)




the_list=[]
layer_with_conv=np.array(the_list)
 #Extract layers from model
for layer in model.layers:
    print(layer)
    print(type(layer))
    if('conv' in layer.name):
        layer_with_conv=np.append(layer_with_conv,layer.name)


full_ds = image_dataset_from_directory(
    data_dir,
    shuffle = False,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

for batch in full_ds:
    print(batch[1])
    for idx in range (1):
        dec_count=0
        for layer in layer_with_conv:

            img=batch[0].numpy()[idx]
            #print(img.shape)
            (Height,Width,Depth)=img.shape
            decrement=2


            NUM_CONV=Height
            NUM_VALS  = 13  #Suits
            NUM_ROTS  = 360 #degrees
            ROT_MOD   = 12  
            ROT_DIV   =  2
            NUM_SUITS =  4
            ROT_LIM   = round(NUM_ROTS/ROT_DIV)
            ROWS_PER_VAL = NUM_SUITS*round(ROT_LIM/ROT_MOD)
            NUM_ROWS  = NUM_VALS * ROWS_PER_VAL

            new_img = np.zeros((NUM_ROWS,NUM_CONV),dtype=np.uint8) #

            layer_target=model.get_layer(layer)

            #dec_count+=1
            the_shape = img.shape
            the_img_arr = np.zeros((1,the_shape[0],the_shape[1], the_shape[2]))

            the_img_arr[0,:,:,:]=img

            activation_model=tf.keras.models.Model(inputs=model.input, outputs=layer_target.output)
            activations=activation_model.predict(the_img_arr)
            activations = activations * 255.0 / 20.0
            activations = np.clip(activations, 0, 255)
            activations = activations.astype(np.uint8)
            print(activations.shape)
            (batch_num ,a_height, a_width, num_depth)=activations.shape

            plot_length=math.ceil(math.sqrt(num_depth))

            fig=plt.figure(figsize=(plot_length,plot_length))
            for r in range(plot_length) :
                for c in range(plot_length) :
                    plt_idx=r*plot_length+c
                    if (plt_idx < num_depth):
                        ax=plt.subplot(plot_length,plot_length,plt_idx+1)
                        a=np.squeeze(activations[0,:,:,plt_idx])
                        print(a.shape)
                        plt.imshow(a)

           
            
            filename="vis_activations_test_layer_%s" % layer
            fig.savefig(filename)
            plt.close(fig)
            break
        break
    break





