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
folder = 'RunE'
folder2 = 'e'
indexN = 1
runName = ['ambepb0','ambepb1','CfPb2','control0','control1','control2','control3'][indexN]
runName2 = ['AmBe Pb 0','ambe pb 1','Cf Pb 2','control 0','control 1','control 2','control 3'][indexN]
filePath = muonVetoPath + os.path.sep + folder +'_'+runName+'_muon.txt'
indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'.txt'
resultsPath = resultsPaths + os.path.sep + folder2 + os.path.sep + runName2+' - Results.txt'
imagesPath = imagesPaths + os.path.sep + folder2 + os.path.sep + runName2 + os.path.sep
imagesFiles = glob.glob(imagesPath+'*.tiff')
imagesFiles = [os.path.basename(i).replace('.tiff','').split('_') for i in imagesFiles]
eventFiles = []
eventPrefixes = []
eventPrefixes.append(imagesFiles[0][0])
eventFiles.append([float(imagesFiles[0][1])])
for i in range(1,len(imagesFiles)):
    if eventPrefixes[-1] != imagesFiles[i][0]:
        eventFiles[-1].append(eventFiles[-1].pop(0))
        eventPrefixes.append(imagesFiles[i][0])
        eventFiles.append([])
    eventFiles[-1].append(float(imagesFiles[i][1]))
eventFiles[-1].append(eventFiles[-1].pop(0))
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
for i in range(len(sNums)):
    groupedTimes.append([times[sNums[i]:eNums[i]]])
    groupedVolts.append([volts[sNums[i]:eNums[i]]])
fig = plt.figure(figsize=(10,5.63))
plt.clf()
plt.clf()
plt.clf()
plt.clf()
before = []
after = []
best = []
for i in range(len(sNums)):
    tmes = np.array(groupedTimes[i])
    vlts = np.array(groupedVolts[i])
    volts1 = vlts[tmes<detectedTimes[i]]
    times1 = tmes[tmes<detectedTimes[i]]
    volts2 = vlts[tmes>detectedTimes[i]]
    times2 = tmes[tmes>detectedTimes[i]]
    volts3 = vlts[eventFiles[i][np.max([0,detectedFrames[i]-20])]<tmes]
    times3 = tmes[eventFiles[i][np.max([0,detectedFrames[i]-20])]<tmes]
    volts3 = volts3[times3<eventFiles[i][np.min([199,detectedFrames[i]+19])]]
    times3 = times3[times3<eventFiles[i][np.min([199,detectedFrames[i]+19])]]
    if volts1[-1]+volts2[0] < 1:
        prevFallingEdge = times1[volts1 > 0.5][-1]
        volts1 = volts1[times1 < prevFallingEdge]
        times1 = times1[times1 < prevFallingEdge]
        before.append(detectedTimes[i]-times1[volts1<0.5][-1])
        try:
            after.append(times2[volts2>0.5][0]-detectedTimes[i])
        except:
            after.append(detectedFrames[i]+100)
        # ,volts1[times1==times1[volts1>0.5][-1]][-1],,volts2[times2==times2[volts2>0.5][0]][0])
    else:
        nextFallingEdge = times2[volts2 < 0.5][0]
        volts2 = volts2[times2 > nextFallingEdge]
        times2 = times2[times2 > nextFallingEdge]
        before.append(detectedTimes[i]-times1[volts1<0.5][-1])
        after.append(times2[volts2>0.5][0]-detectedTimes[i])
        # print(detectedTimes[i]-times1[volts1<0.5][-1],volts1[times1==times1[volts1<0.5][-1]][-1],times2[volts2<0.5][0]-detectedTimes[i],volts2[times2==times2[volts2<0.5][0]][0])
    # print(times[times>detectedTimes[i]][volts[times>detectedTimes[i]]>0.5][0])
    if before[i] < after[i]:
        best.append(-before[i])
    else:
        best.append(after[i])
    # groupedTimes.append(times[sNums[i]:eNums[i]])
    # groupedVolts.append(volts[sNums[i]:eNums[i]])
    # print(eventFiles[i][np.min([199,detectedFrames[i]])])
    # print(detectedTimes[i])
    subPlot = plt.subplot(int(np.ceil(np.sqrt(len(sNums)))),int(np.ceil(np.sqrt(len(sNums)))),i+1)
    plt.vlines(eventFiles[i][np.min([199,detectedFrames[i]])],0,1,colors='r')
    plt.plot(times3,volts3)
    plt.vlines(eventFiles[i][np.max([0,detectedFrames[i]-20]):np.min([200,detectedFrames[i]+20]):1],0,.5,colors='black')
    plt.vlines(eventFiles[i][np.min([199,detectedFrames[i]])],0,1,colors='r')
    plt.vlines(detectedTimes[i] - before[i],0,1,colors='green')
    plt.vlines(detectedTimes[i] + after[i],0,1,colors='orange')
    plt.xlim(eventFiles[i][np.max([0,detectedFrames[i]-20])],eventFiles[i][np.min([199,detectedFrames[i]+19])])
    plt.ylim(-.2,1.5)
    plt.xticks([])
    plt.yticks([])
    subPlot.text(eventFiles[i][np.max([1,detectedFrames[i]-9])],1,str(i+1),fontsize=8)
plt.suptitle(folder2+' - '+runName2+' - MuonVeto Overlay (N='+str(len(sNums))+')')
fig.tight_layout()
plt.savefig(folder2+' - '+runName2+' - Muon.jpg')
# plt.clf()
# plt.show(block=True)
deltas = [before,after,best]
fig = plt.figure(figsize=(10,5.63))
for i in range(2,3):
    # plt.subplot(1,3,i+1,title=['Future Rising Edge Delta t\'s','Past Rising Edge Delta t\'s','Minimum Rising Edge Delta t\'s'][i])
    counts,bins = np.histogram(deltas[i],20,range=(-1,1))
    plt.hist(bins[:-1],bins,weights=counts/np.sum(counts))
plt.suptitle(folder2+' - '+runName2+' - MuonVeto Delta t\'s (N='+str(len(sNums))+')')
plt.savefig(folder2+' - '+runName2+' - Muon Hist.jpg')
# plt.clf()