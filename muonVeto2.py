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
indexN = 2
fileQ = ['c*tr*l','ambe','ambe*pb','cs*137','fiesta','UBe','cf*pb'][indexN]
name = ['control', 'AmBe','AmBe Pb', 'Cs137', 'fiesta','UBe','CfPb'][indexN]
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
binsPerSecond = 50
for fileName in runNameNames:
    thisFile = open(fileName,'r')
    thisData = thisFile.read()
    thisFile.close()
    data += [float(i) for i in thisData.split('\n')]
fig = plt.figure(figsize=(10,5.63))
counts,bins = np.histogram(data,12*binsPerSecond,range=(-6,6))
plt.hist(bins[:-1],bins,weights=counts/np.sum(counts))
plt.suptitle(name+' Ag - MuonVeto Delta t\'s (N='+str(len(data))+')')
plt.savefig(name+' Ag - Muon Hist.jpg')
txtFile = open(name+' Ag - Muon Delta T (best).txt','w')
txtFile.write('\n'.join([str(i) for i in data]))
txtFile.close()

# plt.clf()