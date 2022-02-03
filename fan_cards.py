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

(H , W) = 4096, 6000
RADIUS = 1200
CX = 2500
CY = 2*H/4   # start near the bottom

# Blank image with RGBA = (0, 0, 0, 0)
composite = np.zeros((H, W, 3), np.uint8)

cardwid = 877
cardhgt = 1172
orgx    = 12
orgy    = 45
# Standard order is Spades, Diamonds, Clubs, Hearts
std_order = (2, 3, 0, 1)
suits = ( 'c', 'h', 's', 'd' )
cardfile = args.cardfile

im = cv2.imread(cardfile)

for suit_idx in range(4):

    suit = std_order[3-suit_idx]
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

        # get an image of the card, by indexing into the master image
        im_card = im[orgy + suit*cardhgt:orgy + (suit+1)*cardhgt, orgx + val*cardwid:orgx + (val+1)*cardwid, :]


        im_mask = np.zeros((cardhgt+2, cardwid+2, 1), dtype=np.uint8)
                
        # generate the mask by flooding the area around the rounded rectangular black border of each card!
        # use the card image, but flood only the mask plane
        cv2.floodFill(image=im_card, seedPoint=(1,1), newVal=255, flags=8|(255 << 8)| cv2.FLOODFILL_MASK_ONLY, mask=im_mask)
        
        # part to keep is now 255, border is now 0
        im_mask = 255-im_mask

        # The diagonal size of the card is sqrt(hgt^2 + wid^2).  This should be the new card size!
        card_sz = int(np.sqrt(cardhgt**2 + cardwid**2))
        im_card_walpha = np.zeros((card_sz, card_sz,4), dtype=np.uint8) 
        im_card_walpha[(card_sz-cardhgt)//2:(card_sz-cardhgt)//2+cardhgt,(card_sz-cardwid)//2:(card_sz-cardwid)//2+cardwid,0:3] = im_card
        im_card_walpha[(card_sz-cardhgt)//2:(card_sz-cardhgt)//2+cardhgt,(card_sz-cardwid)//2:(card_sz-cardwid)//2+cardwid,  3] = im_mask[ 1:cardhgt+1, 1:cardwid+1, 0 ]
        
        # angle goes from -90 to 90 degrees
        angle = (suit_idx*13+val)/51 * 180 - 90

        fn = (val_dir + "/card_" + suit_str + "_" + val_str + "_%03d_deg" + ".png") % angle

        M = cv2.getRotationMatrix2D((card_sz/2,card_sz/2),-1*angle,1)
        #        cv.warpAffine( src,           M, dest_size_tuple )
        im_rot = cv2.warpAffine(im_card_walpha,M,(card_sz,card_sz))

        alpha_s = im_rot[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        angle_rad = angle / 360 * (2*np.pi)
        x_offset = int(CX + np.sin(angle_rad) * RADIUS*1.6)
        y_offset = int(CY - np.cos(angle_rad) * RADIUS*1.2)                 # negative is upward
        y1, y2 = y_offset, y_offset + im_rot.shape[0]
        x1, x2 = x_offset, x_offset + im_rot.shape[1]

        # print("x1=%d, x2=%d, y1=%d, y2=%d" % (x1, x2, y1, y2))
        # print("card_sz              =%d"    % card_sz)
        # print("im_card.shape        =%20s"  % str(im_card.shape))
        # print("im_card_walpha.shape =%20s"  % str(im_card_walpha.shape))
        # print("im_rot.shape         =%20s"  % str(im_rot.shape))
        # print("alpha_s.shape        =%20s"  % str(alpha_s.shape))
        # print("alpha_l.shape        =%20s"  % str(alpha_l.shape))

        for c in range(0, 3):
            composite[y1:y2, x1:x2, c] = (alpha_s * im_rot[:, :, c] +
                                      alpha_l * composite[y1:y2, x1:x2, c])

        #im_rot = cv2.resize(im_rot, (256,256))
        #cv2.imwrite( fn, im_rot )

        #fn = "card_" + suit_str + "_" + val_str + "_mask" + ".png"
        #cv2.imwrite( fn, im_mask )


cv2.imwrite("outfile.png", composite)

