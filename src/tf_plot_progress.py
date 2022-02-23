import sys
import os
import re
import subprocess
import numpy as np
import argparse
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

parser = argparse.ArgumentParser()
parser.add_argument('-k',      action='store_true', help='add plot legend')
parser.add_argument('-t',      action='store_true', help='add PyPlot interactive toolbar')
parser.add_argument('logfile', help='path of input file, e.g. ~/exps/resnet50/checkpoint/progress.log' )
parser.add_argument('--title',
                    metavar = '<String>',
                    help = "Add title",
                    required = False,
                    type = str)
    
args = parser.parse_args()

# Usage: log2csv.py -s -tstl -tsta -trnl -trna <pattern> <csv_filename_root>

if (args.t == 0):
    mpl.rcParams['toolbar'] = 'None'


mode_lgnd       = args.k
logfile         = args.logfile
title           = args.title

fd = open( logfile, "r" )

#tsta = np.array([])
tsta = np.empty([0,2])
tstl = np.empty([0,2])
trna = np.empty([0,2])
trnl = np.empty([0,2])

iter_num = 0

# Epoch 1/500
# 232/232 - 10s - loss: 1.3187 - accuracy: 0.4811 - val_loss: 3.5967 - val_accuracy: 0.0969
# Epoch 2/500
# 232/232 - 5s - loss: 0.6243 - accuracy: 0.7594 - val_loss: 7.6614 - val_accuracy: 0.1249
# Epoch 3/500

tt = np.ndarray( [1,2] )

while (1):
    x = fd.readline()
    if x == "":
        break

    #print(x)
    pattern = "Epoch\s+([0-9]+)/"
    m = re.search( pattern, x )
    if m:
        iter_num = int(m.group(1))
        #print( "iter_num=%d" % iter_num )
    
    pattern = "val_accuracy:\s+([0-9\.e+\-]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        tt[:,:] = [iter_num, val]
        tsta=np.append(tsta, tt, 0 )
    
    pattern = "val_loss:\s+([0-9\.e+\-]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        tt[:,:] = [iter_num, val]
        tstl=np.append(tstl, tt, 0)
    
    pattern = "\s+accuracy:\s+([0-9\.e+\-]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "train accu = %f" % val )
        tt[:,:] = [iter_num, val]
        trna=np.append(trna, tt, 0)
    
    pattern = "\s+loss:\s+([0-9\.e+\-]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "str = %s, train loss = %12.9f" % (m.group(1),val) )
        tt[:,:] = [iter_num, val]
        trnl=np.append(trnl, tt, 0)

#num_epoch = max( len(tsta), len(tstl), len(trna), len(trnl) )
num_epoch = max( len(trna), len(trnl) )

plt.rcParams["figure.figsize"] = (12,8)

ax1 = plt.subplot(211)
ax1.plot(trna[:,0], trna[:,1], 'o--')
ax1.plot(tsta[:,0], tsta[:,1], 'o-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.tick_params('y')
plt.ylim((0.0, 1.1))
if title != None:
    plt.title(title)


if mode_lgnd:
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.3, 0.3), shadow=True)

ax2 = plt.subplot(212, sharex=ax1)
ax2.semilogy(trnl[:,0], trnl[:,1], 'o--')
ax2.semilogy(tstl[:,0], tstl[:,1], 'o-')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.tick_params('y')

plt.tight_layout()
plt.show()


