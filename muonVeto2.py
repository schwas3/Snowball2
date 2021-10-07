 #!~/.anaconda3/bin/python
import math
from PIL.Image import new
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import cv2
import glob
# from os import read, startfile, write
# from os.path import exists
# from numpy.core.fromnumeric import reshape, shape
# from numpy.core.numeric import zeros_like
# from numpy.lib.function_base import rot90

muonVetoPath = 'C:\\Users\\Scott\\Downloads\\ForScott'
muonVetoIndicesPath = 'C:\\Users\\Scott\\Downloads\\ForScott2'
resultsPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball6\\Bar and Hist Figs'
imagesPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball9\\SNOWBALL CROPPED IMAGES'
folder = 'RunE'
folder2 = 'e'
runName = ['ambepb0','ambepb1','CfPb2','control0','control1','control2','control3'][2]
runName2 = ['AmBe Pb 0','ambe pb 1','Cf Pb 2','control 0','control 1','control 2','control 3'][2]
filePath = muonVetoPath + os.path.sep + folder +'_'+runName+'_muon.txt'
indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'.txt'
resultsPath = resultsPaths + os.path.sep + folder2 + os.path.sep + runName2+' - Results.txt'
imagesPath = imagesPaths + os.path.sep + folder2 + os.path.sep + runName2 + os.path.sep
# imagesFiles = glob.glob(imagesPath+'*.tiff')
# imagesFiles = [os.path.basename(i).replace('.tiff','').split('_') for i in imagesFiles]
# eventFiles = []
# eventPrefixes = []
# eventPrefixes.append(imagesFiles[0][0])
# eventFiles.append([float(imagesFiles[0][1])])
# for i in range(1,len(imagesFiles)):
#     if eventPrefixes[-1] != imagesFiles[i][0]:
#         eventFiles[-1].append(eventFiles[-1].pop(0))
#         eventPrefixes.append(imagesFiles[i][0])
#         eventFiles.append([])
#     eventFiles[-1].append(float(imagesFiles[i][1]))
# eventFiles[-1].append(eventFiles[-1].pop(0))
txtFile = open(filePath,'r')
indFile = open(indexPath,'r')
resFile = open(resultsPath,'r')
txtLines = txtFile.read()
indLines = indFile.read()
resLines = resFile.read()
txtLines = txtLines.replace('\t','\n').split('\n')
indLines = indLines.replace('\t','\n').split('\n')
resLines = resLines.replace('_','\n').replace(',','\n').split('\n')
txtLines.pop()
indLines.pop()
resLines.pop()
times = [float(i) for i in txtLines[::2]]
volts = [float(i) for i in txtLines[1::2]]
detectedFrames = [int(i) for i in resLines[0::3]]
detectedTimes = [float(i) for i in resLines[2::3]]
sNums = [int(i)-1 for i in indLines[23::11]]
eNums = [int(i) for i in indLines[24::11]]
groupedTimes = []
groupedVolts = []
# fig = plt.figure(figsize=(10,5.63))
# plt.clf()
for i in range(len(sNums)):
    # subPlot = plt.subplot(int(np.ceil(np.sqrt(len(sNums)))),int(np.ceil(np.sqrt(len(sNums)))),i+1)
    groupedTimes.append(times[sNums[i]:eNums[i]])
    groupedVolts.append(volts[sNums[i]:eNums[i]])
    # plt.vlines(eventFiles[i][np.min([199,detectedFrames[i]])],0,1,colors='r')
    # plt.plot(groupedTimes[-1],groupedVolts[-1])
    # plt.vlines(eventFiles[i][np.max([0,detectedFrames[i]-20]):np.min([200,detectedFrames[i]+20]):1],0,.5,colors='black')
    # plt.vlines(eventFiles[i][np.min([199,detectedFrames[i]])],0,1,colors='r')
    # plt.xlim(eventFiles[i][np.max([0,detectedFrames[i]-20])],eventFiles[i][np.min([199,detectedFrames[i]+19])])
    # plt.ylim(-.2,1.5)
    # plt.xticks([])
    # plt.yticks([])
    # subPlot.text(eventFiles[i][np.max([1,detectedFrames[i]-9])],1,str(i+1),fontsize=8)
# plt.suptitle(folder2+' - '+runName2+' - MuonVeto Overlay (N='+str(len(sNums))+')')
# fig.tight_layout()
# plt.savefig(folder2+' - '+runName2+' - Muon.jpg')
