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
# resultsPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball2\\Bar and Hist Figs'
imagesPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball8\\SNOWBALL CROPPED IMAGES'
folder2 = 'E'
# folder2 = 'Run06'
folder = 'Run'+folder2
# folder = folder2[:3]+folder2[3].replace('0','')+folder2[4:]
runName2 = 'Cf Pb 2'
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
# print(folder)
# print(runName2)
# print('[0] = Event Number')
# print('[1] = Time of last frame minus time of first frame')
# print('[2] = Number of frames in event')
# print('[3] = [2] divided by [1]')
# print('[0]\t[1]\t[2]\t[3]')
filePath = muonVetoPath + os.path.sep + folder +'_'+runName+'_muon.txt'
indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'.txt'
if (folder2 == 'Run01' and runName2 == 'ambe_pb') or (folder2 == 'Run02' and runName2 == 'ambe'):
    indexPath = muonVetoIndicesPath + os.path.sep + folder + '_'+runName+'1.txt'
resultsPath = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball2\\E\\'+runName2+' - Results.txt'
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
        # print(str(len(eventFiles))+'\t'+str(round(eventFiles[-1][-2]-eventFiles[-1][0],4))+'\t'+str(len(eventFiles[-1]))+'\t'+str(round((len(eventFiles[-1])-2)/(eventFiles[-1][-2]-eventFiles[-1][0]),4)))
        eventPrefixes.append(imagesFiles[i][0])
        eventFiles.append([])
        invalid.append(False)
    try:
        eventFiles[-1].append(float(imagesFiles[i][1]))
    except:
        invalid[-1] = True
eventFiles[-1].append(eventFiles[-1].pop(0))
txtFile = open(filePath,'r')
indFile = open(indexPath,'r')
resFile = open(resultsPath,'r')
txtLines = txtFile.read()
indLines = indFile.read()
resLines = resFile.read()
txtFile.close()
indFile.close()
resFile.close()
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
    groupedTimes.append(times[sNums[i]:eNums[i]])
    groupedVolts.append(volts[sNums[i]:eNums[i]])
fig = plt.figure(figsize=(10,5.63))
plt.clf()
plt.clf()
plt.clf()
plt.clf()
if folder2=='e' and runName2 == 'ambe pb 0':
    detectedTimes.pop(0)
    detectedFrames.pop(0)
    eventPrefixes.pop(0)
    eventFiles.pop(0)
    invalid.pop(0)
before = []
after = []
best = []
# print(len(sNums))
# print(len(detectedTimes))
# print(len(sNums)==len(eNums)==len(detectedFrames)==len(detectedTimes)==len(eventPrefixes)==len(eventFiles))
risingEdges = 0
timePassed = 0
for i in range(len(sNums)):
    print(i)
    tmes = np.array(groupedTimes[i])
    vlts = np.array(groupedVolts[i])
    vlts1 = vlts[:-1]
    vlts2 = vlts[1:]
    risingEdges += np.sum(vlts2-vlts1>0.5)
    timePassed += tmes[-1]-tmes[0]
    volts1 = vlts[tmes<detectedTimes[i]]
    times1 = tmes[tmes<detectedTimes[i]]
    volts2 = vlts[tmes>detectedTimes[i]]
    times2 = tmes[tmes>detectedTimes[i]]
    volts3 = vlts[eventFiles[i][np.max([0,detectedFrames[i]-20])]<tmes]
    times3 = tmes[eventFiles[i][np.max([0,detectedFrames[i]-20])]<tmes]
    volts3 = volts3[times3<eventFiles[i][np.min([len(eventFiles[i])-1,detectedFrames[i]+19])]]
    times3 = times3[times3<eventFiles[i][np.min([len(eventFiles[i])-1,detectedFrames[i]+19])]]
    if (len(volts1) > 0 and volts1[-1] < 0.5) or (len(volts2) > 0 and volts2[0] < 0.5):
        try:
            prevFallingEdge = times1[volts1 > 0.5][-1]
            volts1 = volts1[times1 < prevFallingEdge]
            times1 = times1[times1 < prevFallingEdge]
            before.append(detectedTimes[i]-times1[volts1<0.5][-1])
        except:
            before.append(10000)
        try:
            after.append(times2[volts2>0.5][0]-detectedTimes[i])
        except:
            after.append(10000)
        # ,volts1[times1==times1[volts1>0.5][-1]][-1],,volts2[times2==times2[volts2>0.5][0]][0])
    else:
        try:
            nextFallingEdge = times2[volts2 < 0.5][0]
            volts2 = volts2[times2 > nextFallingEdge]
            times2 = times2[times2 > nextFallingEdge]
            after.append(times2[volts2>0.5][0]-detectedTimes[i])
        except:
            after.append(10000)
        try:
            before.append(detectedTimes[i]-times1[volts1<0.5][-1])
        except:
            before.append(10000)
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
    plt.plot(times3,volts3,zorder = 1)
    plt.vlines(eventFiles[i][np.max([0,detectedFrames[i]-5]):np.min([200,detectedFrames[i]+5]):1],0,.5,colors='black',zorder = 0)
    plt.vlines(eventFiles[i][np.max([0,detectedFrames[i]-20]):np.min([200,detectedFrames[i]+20]):5],0,.5,colors='black',zorder = 0)
    plt.vlines(eventFiles[i][np.min([len(eventFiles[i])-1,detectedFrames[i]])],.5,1,colors='r',zorder=10)
    plt.vlines(detectedTimes[i] - before[i],0,1,colors='green',zorder = 5)
    plt.vlines(detectedTimes[i] + after[i],0,1,colors='orange',zorder = 4)
    plt.vlines(detectedTimes[i] + best[i],.5,1,colors='yellow',zorder = 6)
    plt.xlim(eventFiles[i][np.max([0,detectedFrames[i]-20])],eventFiles[i][np.min([len(eventFiles[i])-1,detectedFrames[i]+19])])
    plt.ylim(-.2,1.5)
    plt.xticks([])
    plt.yticks([])
    subPlot.text(eventFiles[i][np.max([1,detectedFrames[i]-9])],1,str(i+1)+' - '+str(round(best[i],4)),fontsize=8)
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
txtFile = open(folder2+' - '+runName2+' - Muon Delta T (best).txt','w')
txtFile.write('\n'.join([str(i) for i in best]))
txtFile.close()
print(risingEdges,timePassed,risingEdges/timePassed)
print(len(sNums)==len(eNums)==len(detectedFrames)==len(detectedTimes)==len(eventPrefixes)==len(eventFiles))
# plt.clf()