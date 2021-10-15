 #!~/.anaconda3/bin/python
import math
from PIL.Image import new
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import cv2
import glob

from numpy.core.shape_base import block
# from os import read, startfile, write
# from os.path import exists
# from numpy.core.fromnumeric import reshape, shape
# from numpy.core.numeric import zeros_like
# from numpy.lib.function_base import rot90

muonVetoPath = 'C:\\Users\\Scott\\Downloads\\ForScott'
muonVetoIndicesPath = 'C:\\Users\\Scott\\Downloads\\ForScott2'
resultsPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball6\\Bar and Hist Figs'
imagesPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball9\\SNOWBALL CROPPED IMAGES'
folder2 = 'E'
folder2 = 'Run09'
folder = 'Run'+folder2
folder = folder2[:3]+folder2[3].replace('0','')+folder2[4:]
runName2 = 'AmBe-blueLED'
runName = runName2.replace(' ','').replace('-','').replace('sss','').replace('pffb','').replace('_','')
# runName = 'ambe'

# NM = number mismatch
# EM = events missing (NM but only several short)
# + = A LOT IS WRONG
# * = Significant outlier data possible
# I = Included
# NI = Not Included

# Cf Pb
# I - e Cf Pb 2
# NI - 

# UBe
# I - 10 UBe_thick_front_blue, 11 UBe_thick_front_blue (EM**)
# NI - 

# fiesta (*)
# I - 07 Fiesta Front (**)
# NI - 08 Fiesta Be Front (junk) (NM/EM++)

# cs137
# I 05 cs-137, 06 cs-137 (last event excluded successfully)
# NI - 04 Cs-137 (NM)

# ambe pb
# I - e ambe pb 0, e ambe pb 1, 01 ambe_pb (*?)
# NI - 00 ambe_pb (NM+)

# ambe
# I - a ambe top, b ambe1, c ambe0, c ambe1, d ambe s 0, d ambe s 1, 02 ambe (*****) 03 ambe (*) 09 ambe-blueled, 12 ambe_side, 13 ambe_side, 14 ambe-top
# NI - a ambe-side (NM+), b ambe0 (NM)

# control
# I a control0, a control1, a control2, a control3, b control0, b control1, c control0, c control1, d control 0, d control 1, e control 0, e control 1, e control 2, e control 3, 01 control, 02 control (*), 03 control, 04 control (*), 06 control, 07 Control, 08 control08, 09 ctrl, 10 ctrl, 11 control, 12 control, 14 ctrl
# NI - 00 control (NM+), 05 control (+), 13 ctrl (NM/EM+)
print(folder)
print(runName2)
print('[0] = Event Number')
print('[1] = Time of last frame minus time of first frame')
print('[2] = Number of frames in event')
print('[3] = [2] divided by [1]')
print('[0]\t[1]\t[2]\t[3]')
filePath = muonVetoPath + os.path.sep + folder +'_'+runName+'_muon.txt'
indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'.txt'
if (folder2 == 'Run01' and runName2 == 'ambe_pb') or (folder2 == 'Run02' and runName2 == 'ambe'):
    indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'1.txt'
resultsPath = resultsPaths + os.path.sep + folder2 + os.path.sep + runName2+' - Results.txt'
imagesPath = imagesPaths + os.path.sep + folder2 + os.path.sep + runName2 + os.path.sep
imagesFiles = glob.glob(imagesPath+'*.tiff')
imagesFiles = [os.path.basename(i).replace('.tiff','').split('_') for i in imagesFiles]
eventFiles = []
eventPrefixes = []
invalid = []
invalid.append(False)
eventPrefixes.append(imagesFiles[0][0])
try:
    eventFiles.append([float(imagesFiles[0][1])])
except:
    invalid[-1] = True
    eventFiles.append([float(imagesFiles[0][1].replace('X',''))])
for i in range(1,len(imagesFiles)):
    if eventPrefixes[-1] != imagesFiles[i][0]:
        eventFiles[-1].append(eventFiles[-1].pop(0))
        print(str(len(eventFiles))+'\t'+str(round(eventFiles[-1][-2]-eventFiles[-1][0],4))+'\t'+str(len(eventFiles[-1]))+'\t'+str(round((len(eventFiles[-1])-2)/(eventFiles[-1][-2]-eventFiles[-1][0]),4)))
        eventPrefixes.append(imagesFiles[i][0])
        eventFiles.append([])
        invalid.append(False)
    try:
        eventFiles[-1].append(float(imagesFiles[i][1]))
    except:
        invalid[-1] = True
eventFiles[-1].append(eventFiles[-1].pop(0))