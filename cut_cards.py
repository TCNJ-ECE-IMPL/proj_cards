#!/usr/bin/python

import sys
import os
import re
import subprocess
import numpy as np
import argparse
import cv2
import errno

parser = argparse.ArgumentParser()
parser.add_argument('cardfile',   help='path of input file, e.g. Color_52_Faces_v.2.0.png' )
args = parser.parse_args()

cardwid = 877
cardhgt = 1172
orgx    = 12
orgy    = 45
suits = ( 'c', 'h', 's', 'd' )
cardfile = args.cardfile

im = cv2.imread(cardfile)


for suit in range(4):

    suit_str = suits[suit]

    for val in range(13):
        
        if val == 0:
            val_str = 'a'
        elif val == 10:
            val_str = 'j'
        elif val == 11:
            val_str = 'q'
        elif val == 12:
            val_str = 'k'
        else:
            val_str = "%d" % (val+1)

        val_dir = "cardval_" + val_str
        try:
            os.makedirs(val_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        im_card = im[orgy + suit*cardhgt:orgy + (suit+1)*cardhgt, orgx + val*cardwid:orgx + (val+1)*cardwid, :]

        im_mask = np.zeros((cardhgt+2, cardwid+2, 1), dtype=np.uint8)
                
        cv2.floodFill(image=im_card, seedPoint=(1,1), newVal=255, flags=8|(255 << 8)| cv2.FLOODFILL_MASK_ONLY, mask=im_mask)
        
        im_mask = 255-im_mask
        im_card_walpha = np.zeros((cardhgt, cardhgt,4), dtype=np.uint8) 
        im_card_walpha[:,(cardhgt-cardwid)//2:(cardhgt-cardwid)//2+cardwid,0:3] = im_card
        im_card_walpha[:,(cardhgt-cardwid)//2:(cardhgt-cardwid)//2+cardwid,3] = im_mask[ 1:cardhgt+1, 1:cardwid+1, 0 ]
        
        for angle in range (0,360,1):
            
            fn = (val_dir + "/card_" + suit_str + "_" + val_str + "_%03d_deg" + ".png") % angle

            M = cv2.getRotationMatrix2D((cardhgt/2,cardhgt/2),angle,1)
            im_rot = cv2.warpAffine(im_card_walpha,M,(cardhgt,cardhgt))

            im_rot = cv2.resize(im_rot, (256,256))
            cv2.imwrite( fn, im_rot )

        #fn = "card_" + suit_str + "_" + val_str + "_mask" + ".png"
        #cv2.imwrite( fn, im_mask )


