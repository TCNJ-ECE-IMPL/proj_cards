
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
import os
import os.path
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import save_model, load_model, model_from_json
from vis.utils import utils


parser=argparse.ArgumentParser(description='Visualizes convolution kernels')

parser.add_argument('--model',metavar='model',type=str, required=True, help='Example: --model <name of model>.json')
parser.add_argument('--wgt_path',metavar='wgt_path',type=str, required=True, help='Example: --wgt_path <weights file>.h5')
parser.add_argument('--file_path',metavar='--file_path',type=str,required=True,help='Example: --file_path<')


args=parser.parse_args()
path=args.file_path

def move_to_path(Path):
    #create directory 
    cwd=os.getcwd()
    new_path=cwd+'/'+Path
   
    if(os.path.isdir(new_path)):
        os.chdir(new_path)
    else:
        cmd='mkdir -p '+Path
        os.system(cmd)
        os.chdir(new_path)



#load model
model_name=args.model

print("Loading " + model_name)

json_file = open(model_name, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

print("Model loaded...")

#load weights
best_wgt_path=args.wgt_path

print("Loading " + best_wgt_path)
model.load_weights(best_wgt_path)
print("Weights loaded onto model...")

print(type(model))

model.summary()
move_to_path(path)  #move to path 

the_list=[]
layer_with_conv=np.array(the_list)
#Extract layers from model
for layer in model.layers:
    if('conv' in layer.name or layer.name in path):
        layer_idx = utils.find_layer_idx(model, layer.name)
        layer_with_conv=np.append(layer_with_conv,layer_idx)


print(layer_with_conv)








gap = 0.2

for l in layer_with_conv:


    weights=model.layers[int(l)].get_weights()
    #(kern_hgt, kern_wd)=weights.size(
    #print(weights)

    weights=np.asarray(weights[0])

    weights_shape = weights.shape
    print("weights shape = " + str(weights_shape))

# The shape of the kernel array is: (KW, KH, IC, OC), where:
#   KW is the kernel width
#   KH is the kernel height
#   IC is the number of input channels
#   OC is the number of output channels
    KH = weights_shape[0]
    KW = weights_shape[1]
    PI = weights_shape[2]
    PO = weights_shape[3]

    #print("Shape of weights is: %s, this represents [PI][KH][KW][PO], where PI=# planes in, KH/KW=kernel wid/hgt, PO=# planes out" % str(weights_shape))
    #print(weights[0])   # [1] are biases, [0] are convolutional weights

    # print(kernel_hgt)
    # print(kernel_wd)


    TILE_ROWS = PO          # 1 row for each output feature map
    TILE_COLS = PI          # 1 column for each input feature map

    NUM_TILES=TILE_ROWS*TILE_COLS       # MUST BE NUMBER OF FEATURE MAPS(Look at weights[0].shape)

    count=0
    
   
    #fig=plt.figure(figsize=(10,320), dpi=100)
    fig=plt.figure(figsize=(KW*TILE_COLS + (TILE_COLS-1)*gap,KH*TILE_ROWS + (TILE_ROWS-1)*gap), dpi=100)
    #fig=plt.figure()
    for r in range(TILE_ROWS):
        for c in  range(TILE_COLS):
            ax=plt.subplot(TILE_ROWS,TILE_COLS,count+1)
            # ax.set_xlim(0,940)
            # ax.set_ylim(1200,9200)
            ax.set_xticks([])
            ax.set_yticks([])
            filt=weights

            f=np.squeeze(filt[:,:,c,r])
            mx =  np.abs(f).max()
            f = f / mx / 2                      # normalize -0.5 to 0.5
            f = f + 0.5                         # normalize -0.5 to 0.5
    #        print("Feature map: "+str(f))
            #plt.tight_layout()
            #plt.tight_layout()
            plt.subplots_adjust(wspace=gap, hspace=gap)
            plt.imshow(f, cmap='gray')
            count=count+1
            #print(count)
          

    #plt.show()
    filename='dcnn_vis_layer_%d.png'% (l)
    fig.savefig(filename)
    print(fig)
    plt.close(fig)

