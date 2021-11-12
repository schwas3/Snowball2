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
thisPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball2'
imagesPaths = 'C:\\Users\\Scott\\Documents\\GitHub\\Snowball9\\SNOWBALL CROPPED IMAGES'
indexN = 3
fileQ = ['c*tr*l','ambe','ambe*pb','cs*137','fiesta','UBe','cf*pb'][indexN]
name = ['control', 'AmBe','AmBe Pb', 'Cs137', 'U','UBe','CfPb'][indexN]
runNamePaths = glob.glob(thisPaths+os.path.sep+ '* - '+fileQ+'* - Muon Delta T (best).txt')
# runNamePaths = glob.glob(resultsPaths + os.path.sep + '*' + os.path.sep + fileQ+'* - Results.txt')
runNameNames = [os.path.basename(i) for i in runNamePaths]
if indexN == 1:
    for i in range(len(runNameNames)):
        storedName = runNameNames.pop(0)
        if storedName.find('pb') == -1:
            runNameNames.append(storedName)
# print(runNameNames)

# runNameNames = [i.split(resultsPaths+os.path.sep)[1] for i in runNamePaths]
# print(runNameNames)
# print(len(runNameNames))
data = []
bigData = []
binsPerSecond = 10
bounds = 1
fig = plt.figure(figsize = (10,5.63))
for i in range(len(runNameNames)):
    fileName = runNameNames[i]
    thisFile = open(fileName,'r')
    thisData = thisFile.read()
    thisFile.close()
    data = [float(i) for i in thisData.split('\n')]
    bigData += data
    subPlot = plt.subplot(int(np.ceil(np.sqrt(len(runNameNames)))),int(np.ceil(np.sqrt(len(runNameNames)))),i+1,title=fileName.split(' - Muon Delta T (best).txt')[0]+' (N='+str(len(data))+')')
    counts,bins = np.histogram(data,int(2*bounds*binsPerSecond),range=(-bounds,bounds))
    plt.plot(bins[:-1]/2+bins[1:]/2,counts/len(data))
    # plt.hist(bins[:-1],bins,weights=counts/len(data))
    plt.ylim(-.05,.6)
plt.suptitle(name+' Separate - MuonVeto Delta t\'s')
fig.tight_layout()
plt.savefig(name+' Separate - Muon Hist.jpg')
fig = plt.figure(figsize=(10,5.63))
counts,bins = np.histogram(bigData,int(2*bounds*binsPerSecond),range=(-bounds,bounds))
plt.plot(bins[:-1]/2+bins[1:]/2,counts/len(bigData))
# plt.errorbar(bins[:-1]/2)
# plt.hist(bins[:-1],bins,weights=counts/len(bigData))
plt.suptitle(name+' Ag - MuonVeto Delta t\'s (N='+str(len(bigData))+')')
plt.savefig(name+' Ag - Muon Hist.jpg')
txtFile = open(name+' Ag - Muon Delta T (best).txt','w')
txtFile.write('\n'.join([str(i) for i in bigData]))
txtFile.close()

# plt.clf()