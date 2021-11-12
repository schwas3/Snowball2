 #!~/.anaconda3/bin/python
import math
from PIL.Image import new
from matplotlib import colors
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
name = ['control', 'AmBe','AmBe Pb', 'Cs137', 'U','UBe','CfPb']
Colors = ['b-','r--','r-','g-','c-','y-','m-','y-']
name = ['control','AmBe Pb', 'CfPb','Cs137']
Colors = ['b-','r-','y-','g-']
# name.pop(2)
# Colors.pop(2)
# runNamePaths = glob.glob(thisPaths+os.path.sep+ '* - '+name+'* - Muon Delta T (best).txt')
# runNamePaths = glob.glob(resultsPaths + os.path.sep + '*' + os.path.sep + fileQ+'* - Results.txt')
# runNameNames = [os.path.basename(i) for i in runNamePaths]
# runNameNames = [i.split(resultsPaths+os.path.sep)[1] for i in runNamePaths]
# print(runNameNames)
# print(len(runNameNames))
data = []
bigData = []
framesPerSecond = 7.4
backgroundFrequency = 0.5
binsPerSecond = framesPerSecond/2
bounds = 2
binWidth = 1 / binsPerSecond
coincedenceRate = binWidth * backgroundFrequency
fig = plt.figure(figsize=(10,5.63))
Counts = []
for i in range(len(name)):
    fileName = name[i]
    thisFile = open(thisPaths+os.path.sep+fileName+' Ag - Muon Delta T (best).txt','r')
    thisData = thisFile.read()
    thisFile.close()
    data = [float(i) for i in thisData.split('\n')]
    bigData.append(data)
    # data = np.array(data)
    # data = data[data<.5]
    # data = data[data>-.5]
    counts,bins = np.histogram(data,int(2*bounds*binsPerSecond),range=(-bounds,bounds))
    if i == 0:
        Counts = counts
# plt.plot(bins[:-1]/2+bins[1:]/2,counts/len(data),Colors[i],label=fileName+' (N='+str(len(data))+')')
    plt.errorbar(bins[:-1]/2+bins[1:]/2,counts/len(bigData[i])-Counts/len(bigData[0]),np.sqrt(counts)/len(bigData[i])+np.sqrt(Counts)/len(bigData[0]),binWidth/2,Colors[i],linewidth=0.25,elinewidth=0.5,label=fileName+' (N='+str(len(data))+')')
    # plt.errorbar(bins[:-1]/2+bins[1:]/2,counts/len(data),np.sqrt(counts)/len(data),binWidth/2,Colors[i],label=fileName+' (N='+str(len(data))+')',linewidth=.5,elinewidth=1)
    plt.hlines([0,0],-bounds,bounds,colors=['black','black'],linestyles=['dashed','dashed'],zorder=0)
plt.xlabel('Delta Time (s)')
plt.ylabel('Probability/bin')
# plt.plot(bins[:-1]/2+bins[1:]/2,counts/np.sum(counts)/int(2*bounds*binsPerSecond),Colors.pop(0),label=fileName+' (N='+str(len(data))+')')
    # plt.hist(bins[:-1],bins,weights=counts/np.sum(counts),label=fileName+' (N='+str(len(data))+')')
plt.legend()
plt.ylim(-.2,.6)
plt.suptitle('ALL Ag - MuonVeto Delta t\'s')
fig.tight_layout()
plt.savefig('ALL Ag - Muon Hist.jpg')
fig = plt.figure(figsize=(10,5.63))
for i in range(len(bigData)):
    fileName = name[i]
    subPlot = plt.subplot(int(np.ceil(np.sqrt(len(bigData)))),int(np.ceil(np.sqrt(len(bigData)))),i+1,title=fileName+' (N='+str(len(bigData[i]))+')')
    counts,bins = np.histogram(bigData[i],int(2*bounds*binsPerSecond),range=(-bounds,bounds))
    plt.plot(bins[:-1]/2+bins[1:]/2,counts/len(bigData[i])-Counts/len(bigData[0]),Colors[i],linewidth=0.25)
    plt.errorbar(bins[:-1]/2+bins[1:]/2,counts/len(bigData[i])-Counts/len(bigData[0]),np.sqrt(counts)/len(bigData[i])+np.sqrt(Counts)/len(bigData[0]),binWidth/2,Colors[i],linewidth=0.25,elinewidth=0.5)
    plt.hlines([0,0],-bounds,bounds,colors=['black','black'],linestyles=['dashed','dashed'],zorder=0)
    plt.ylim(-.2,.6)
    plt.xlabel('Delta Time (s)')
    plt.ylabel('Probability/bin')
plt.suptitle('ALL separate - MuonVeto Delta t\'s')
fig.tight_layout()
plt.savefig('ALL separate - Muon Hist.jpg')
# txtFile = open(name+' Ag - Muon Delta T (best).txt','w')
# txtFile.write('\n'.join([str(i) for i in data]))
# txtFile.close()

# plt.clf()