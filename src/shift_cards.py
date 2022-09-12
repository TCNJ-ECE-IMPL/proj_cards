import sys
import os
import re
import subprocess
import numpy as np
import argparse
import cv2
import errno

        

suits=('c' , 'h' , 's' , 'd')       # create a tuple of suit names

       
for suit_num in range(4):           # step through the 4 suits
    suit=suits[suit_num]            # get the character name of the suit

    for val in range(13):           # step through each card value
                                    # for numbered cards, the index 'val' is one less than a card's value
                                    # 0 -> Ace, 10 -> Jack, 11 -> Queen, 12 -> King
        for angle in range(0,180,4):     
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
            
            # Build a filename that represents the source card's suit, value and rotation
            filepath= "/home/leopole1/proj_cards/cardval_%s/card_%s_%s_%03d_deg.png" % (val_str, suit, val_str ,angle)
           
            # Read in the card source file
            im=cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            (H,W,dim)=np.shape(im)

            blank_im=np.zeros((W,H,4), np.uint8) #use to zoom out,shift, ect
   
            scale=1.25

            dirname="/home/leopole1/proj_cards/augment_card/cardval_%s" % (val_str)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

#zoom in
             #resize the image by scaling, extract dimensions of larger image
            im_ZI=cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            (New_H,New_W,dim)=np.shape(im_ZI)

            cutH=int(0.5*(New_H-H)) #Section between image we want and image of larger size. 
            cutW=int(0.5*(New_W-W)) #The difference is halved as we want to cut off each side one by one.
        
            
            #remove sections of the image, cropping it back down to its original sizeS
            im_ZI=np.delete(im_ZI,slice(0,cutW,1),axis=1) #left side
            im_ZI=np.delete(im_ZI,slice(W,New_W,1),axis=1) #right side
            im_ZI=np.delete(im_ZI,slice(0,cutH,1),axis=0) #top
            im_ZI=np.delete(im_ZI,slice(H,New_H,1),axis=0) #bottom
         

         

            shift=20   
            #iterate through the nine possible values and copy the array depending on direction horizontally and vertically
   
            h_labels=("hl","hc","hr")
            v_labels=("vu","vc","vd")

            for h in range(3):
                h_str=h_labels[h]
                if h==0:
                    W_start=0   #Detirmine start and end values to copy im_ZI into a blank composite
                    W_end=W-1-shift
                    WC_start=shift
                    WC_end=W-1
                elif h == 1:
                    W_start=0
                    W_end=W-1
                    WC_start=0
                    WC_end=W-1
                else: #h=2
                    W_start=shift
                    W_end=W-1
                    WC_start=0
                    WC_end=W-1-shift

                for v in range(3) :
                    v_str=v_labels[v]

                    if v == 0:
                        H_start=0
                        H_end=H-1-shift
                        HC_start=shift
                        HC_end=H-1
                    elif v == 1:
                        H_start=0
                        H_end=H-1
                        HC_start=0
                        HC_end=H-1
                    else :
                        H_start=shift
                        H_end=H-1
                        HC_start=0
                        HC_end=H-1-shift

                blank_im[H_start:H_end,W_start:W_end]=im_ZI[HC_start:HC_end,WC_start:WC_end]

                #write file with movement and zoom included
                filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,h_str+"_"+v_str,'ZOOM1.25')
                cv2.imwrite(os.path.join(dirname,filename),blank_im)
                blank_im=np.zeros((W,H,4), np.uint8) #new blank image


             

           
            


        #zoom out 
            scale=.75
          
            im_ZO=cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) #Zoom out using resize
            (New_H,New_W,dim)=np.shape(im_ZO)   #get dimensions of new image
            
            
         
           

            DiffW=int(.5*(W-New_W))  #detirmine defference between two images-Use 1/2 as we want to work with each side
            DiffH=int(.5*(H-New_H)) 


            #Iterate through for loops for all possible direction combinations. 
            W_shift=0
            H_shift=0

            for h in range(3):
                h_str=h_labels[h]
                if h==0 :
                    W_shift=-shift
                elif h == 1:
                    W_shift = 0
                else:
                    W_shift=shift

                for v in range (3):
                    v_str=v_labels[v]
                    if h==0 :
                        H_shift=-shift
                    elif h == 1:
                        H_shift = 0
                    else:
                        H_shift=shift

                    #Copy the zoomed out image into a blank composite
                    blank_im[DiffH-1+H_shift:H-1-DiffH+H_shift,DiffW-1+W_shift:(W-1-DiffW+W_shift)]=im_ZO[:,:]
                    #write file with movement and zoom included
                    filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,h_str+"_"+v_str,'ZOOM.75')
                    cv2.imwrite(os.path.join(dirname,filename),blank_im)
                    blank_im=np.zeros((W,H,4), np.uint8) #new blank image