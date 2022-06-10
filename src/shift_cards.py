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
        for angle in range(180):     
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

#zoom in
   
            im_ZI=cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            (New_H,New_W,dim)=np.shape(im_ZI)

            cutH=int(.5*(New_H-H))
            cutW=int(.5*(New_W-W))
        

            im_ZI=np.delete(im_ZI,slice(0,cutW,1),axis=1)
            im_ZI=np.delete(im_ZI,slice(W,New_W,1),axis=1)
            im_ZI=np.delete(im_ZI,slice(0,cutH,1),axis=0)
            im_ZI=np.delete(im_ZI,slice(H,New_H,1),axis=0) #slice bottom
         

         
   
            shift=20


            #right
            blank_im[:, shift:W-1]=im_ZI[:,0:W-shift-1]
            


            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vc','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image
           

           #up 
            blank_im[0:H-1-shift,:]=im_ZI[shift:H-1 ,:]
           

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vu','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

           #center
            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vc','ZOOM125%')
            cv2.imwrite(filename,im_ZI)


         
           #left

            blank_im[:,0:W-1-shift]=im_ZI[:,shift:W-1]


            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vc','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)



            #down
            blank_im[shift:H-1,:]=im_ZI[0:H-1-shift, :]



            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vd','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)
         

           #up right
            blank_im[0:H-shift-1:,shift:W-1]=im_ZI[shift:H-1,:W-shift-1]



            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vu','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)


           #down right
            blank_im[shift:H-1,shift:W-1]=im_ZI[0:H-shift-1,0:W-shift-1]



            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vd','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)



           #up left
            blank_im[0:H-1-shift,0:W-1-shift]=im_ZI[shift:H-1 , shift:W-1]





            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vu','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)

          
            #down left
            blank_im[shift:H-1,0:W-1-shift]=im_ZI[0:H-1-shift,shift:W-1]


            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vd','ZOOM125%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8)

            


        #zoom out 
            scale=.75
          
            im_ZO=cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            (New_H,New_W,dim)=np.shape(im_ZO)
            
            #Isolate the card
         
           

            DiffW=int(.5*(W-New_W))
            DiffH=int(.5*(H-New_H))


 

            blank_im[DiffH-1:H-1-DiffH,DiffW-1:(W-1-DiffW)]=im_ZO[:,:]

         #center
            blank_im[DiffH-1:H-1-DiffH,DiffW-1:(W-1-DiffW)]=im_ZO[:,:]

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vc','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image
        

         #move right   
            blank_im[DiffH-1:H-1-DiffH,DiffW-1+shift:(W-1-DiffW+shift)]=im_ZO[:,:]



            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vc','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

          #move left

            blank_im[DiffH-1:H-1-DiffH,DiffW-1-shift:(W-1-DiffW-shift)]=im_ZO[:,:]


            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vc','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

           #move up
            blank_im[DiffH-1-shift:H-1-DiffH-shift,DiffW-1:(W-1-DiffW)]=im_ZO[:,:]
           

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vu','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

           
            #move down

            blank_im[DiffH-1+shift:H-1-DiffH+shift,DiffW-1:(W-1-DiffW)]=im_ZO[:,:]


            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hc_vd','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

            #up right
            blank_im[DiffH-1-shift:H-1-DiffH-shift,DiffW-1+shift:(W-1-DiffW+shift)]=im_ZO[:,:]

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vu','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

            #up left 
            blank_im[DiffH-1-shift:H-1-DiffH-shift,DiffW-1-shift:(W-1-DiffW-shift)]=im_ZO[:,:]

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vu','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

            #down left
            blank_im[DiffH-1+shift:H-1-DiffH+shift,DiffW-1-shift:(W-1-DiffW-shift)]=im_ZO[:,:]

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hl_vd','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image

            #down right
            blank_im[DiffH-1+shift:H-1-DiffH+shift,DiffW-1+shift:(W-1-DiffW+shift)]=im_ZO[:,:]

            filename="card_%s_%s_%03d_deg_%s_%s.png" % (suit,val_str,angle,'hr_vd','ZOOM75%')
            cv2.imwrite(filename,blank_im)
            blank_im=np.zeros((W,H,4), np.uint8) #new blank image


























