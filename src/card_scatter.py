import os
import time
import argparse
import itertools
from datetime import date
from itertools import cycle

import scipy.io as Mat
import numpy as np
import matplotlib.pyplot as plt
import random, sklearn
import cv2


(H , W) = 1080, 1980
# Blank image with RGBA = (0, 0, 0, 0)
composite = np.zeros((H, W, 3), np.uint8)

idx = 0     # index into data array
for i in range(1000):
    for suit in ('s', 'h', 'd', 'c'):
        val = random.randint(1, 13)
        deg = random.randint(0,359)

        if val >= 2 and val <= 10:
            val_char = "%d" % val
        elif val == 1:
            val_char = "a"
        elif val == 11:
            val_char = "j"
        elif val == 12:
            val_char = "q"
        elif val == 13:
            val_char = "k"

        filepath = "/home/pearlstl/cards/source/as_images/cardval_%s/card_%s_%s_%03d_deg.png" % (val_char, suit, val_char, deg)
        s_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED);

        x_offset = np.int32(random.random()*(W-257))
        y_offset = np.int32(random.random()*(H-257))
        y1, y2 = y_offset, y_offset + s_img.shape[0]
        x1, x2 = x_offset, x_offset + s_img.shape[1]

        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            composite[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                      alpha_l * composite[y1:y2, x1:x2, c])
        idx = idx + 1


tf.data.Dataset.from_tensor_slices(train_x)

cv2.imwrite("outfile.png", composite)
