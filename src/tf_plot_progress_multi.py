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

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
parser._action_groups.append(optional)

optional.add_argument('-k',         action='store_true',   help='add plot legend',                   required = False)
optional.add_argument('-l',         action='store_true',   help='loss only',                         required = False)
optional.add_argument('-t',         action='store_true',   help='add PyPlot interactive toolbar',    required = False)
optional.add_argument('--title',    nargs=1, metavar = '<String>',  help = "Add title",                       required = False, default=None, type = str)
    
required.add_argument('logfile',    nargs="+", metavar='<logfile>', type=str,          help='path of input file, e.g. ~/exps/resnet50/checkpoint/progress.log' )

args = parser.parse_args()

# Usage: log2csv.py -s -tstl -tsta -trnl -trna <pattern> <csv_filename_root>

if (args.t == 0):
    mpl.rcParams['toolbar'] = 'None'


mode_lgnd       = args.k
logfiles        = args.logfile
title           = args.title
loss_only       = args.l

plt.rcParams["figure.figsize"] = (12,8)

if loss_only:
    ax_loss_trn = plt.subplot(211)
    ax_loss_tst = plt.subplot(212, sharex=ax_loss_trn)

else:
    ax_loss_trn = plt.subplot(411)
    ax_loss_tst = plt.subplot(412, sharex=ax_loss_trn)
    ax_acc_trn  = plt.subplot(413, sharex=ax_loss_trn)
    ax_acc_tst  = plt.subplot(414, sharex=ax_loss_trn)

for logfile in logfiles:
    fd = open( logfile, "r" )

    print(logfile + ":")

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
            if (val < 1e-10):
                val = 1e-10
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
            if (val < 1e-10):
                val = 1e-10
            #print( "str = %s, train loss = %12.9f" % (m.group(1),val) )
            tt[:,:] = [iter_num, val]
            trnl=np.append(trnl, tt, 0)

    #num_epoch = max( len(tsta), len(tstl), len(trna), len(trnl) )
    num_epoch = max( len(trna), len(trnl) )

    if re.search("NAdam", logfile):

        ax_loss_trn.semilogy(trnl[:,0], trnl[:,1], 'o--')
        ax_loss_tst.semilogy(tstl[:,0], tstl[:,1], 'o-')

        if loss_only != True:
            ax_acc_trn.plot(trna[:,0], trna[:,1], 'o--')
            ax_acc_tst.plot(tsta[:,0], tsta[:,1], 'o-')
    else:

        ax_loss_trn.semilogy(trnl[:,0], trnl[:,1], 'o--')
        ax_loss_tst.semilogy(tstl[:,0], tstl[:,1], 'o-')

        if loss_only != True:
            ax_acc_trn.plot(trna[:,0], trna[:,1], 'o--')
            ax_acc_tst.plot(tsta[:,0], tsta[:,1], 'o-')

    fd.close()

if mode_lgnd:
    plt.legend(logfiles, loc='best', shadow=True)

if title != None:
    plt.title(title)

ax_loss_trn.set_xlabel('Epoch')
ax_loss_trn.set_ylabel('Cross-Entropy Training Loss')
ax_loss_trn.tick_params('y')

ax_loss_tst.set_xlabel('Epoch')
ax_loss_tst.set_ylabel('Cross-Entropy Validation Loss')
ax_loss_tst.tick_params('y')

if loss_only != True:
    ax_acc_trn.set_xlabel('Epoch')
    ax_acc_trn.set_ylabel('Training Accuracy')
    ax_acc_trn.tick_params('y')

    ax_acc_tst.set_xlabel('Epoch')
    ax_acc_tst.set_ylabel('Validation Accuracy')
    ax_acc_tst.tick_params('y')


plt.tight_layout()

plt.show()


