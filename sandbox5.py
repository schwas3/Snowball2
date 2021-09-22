 #!~/.anaconda3/bin/python
import math
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import yaml
import cv2
from PIL import Image
import glob
from os import read, startfile, write
from os.path import exists
def makeBarGraph(labels,width,barLabels,barData,title,yLabel,xLabel, makeBar: bool, makeHist: bool):
    if makeBar:
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(figsize=(10,5.63))
        rects1 = ax.bar(x - width/2, barData[0], width, label = barLabels[0])
        rects2 = ax.bar(x + width/2, barData[1], width, label = barLabels[1])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(yLabel,fontsize = 10)
        ax.set_xlabel(xLabel,fontsize = 10)
        ax.set_title(title,fontsize = 12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,fontsize = 6)
        ax.legend(fontsize = 8)

        ax.bar_label(rects1, padding=3, fontsize = 6)
        ax.bar_label(rects2, padding=3,fontsize = 6)

        fig.tight_layout()
        suffix = 0
        while exists(str(title)+' '+str(suffix)+'.png'):
            suffix += 1
        plt.savefig(str(title)+' '+str(suffix)+'.png')
    if makeHist:
        suffix = 0
        fig = plt.figure(figsize=(10,5.63))
        plt.subplot(1,2,1,title='Code Output' + ' ('+str(title)+')')
        plt.hist(barData[0],bins=8,alpha=0.5,rwidth=0.9)
        plt.xlabel('Detection Frame')
        plt.ylabel('Frequency')
        plt.subplot(1,2,2,title='Answer Key' + ' ('+str(title)+')')
        plt.xlabel('Detection Frame')
        plt.ylabel('Frequency')
        plt.hist(barData[1],bins=8,alpha=0.5,rwidth=0.9)
        while exists(str(title)+' hist '+str(suffix)+'.png'):
            suffix += 1
        plt.savefig(str(title)+' hist '+str(suffix)+'.png')
def padRun(runImages):
    return np.pad(runImages,[(0,0),(1,1),(1,1)],'constant',constant_values = 255)
def initializeImages(runImages):
    newImages = []
    for runImage in runImages:
        newImages.append(runImage)
    return newImages
def runFrameStamp(runNumber,runImages):
    images = initializeImages(runImages)
    for frameNumber in range(len(images)):
        images[frameNumber] = imgNumStamps(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,(frameNumber+1)%len(images)),0,0,images[frameNumber])
    return images
def normalizePixelValues(runImages):
    images = initializeImages(runImages)
    mid = int(np.mean(images))
    bottom = 2*mid-255
    if 255 - mid < mid:
        images = np.where(np.array(images) < bottom, 0,np.multiply(np.subtract(images,bottom),255/(255-bottom)))
    else:
        top = 2*mid
        images = np.where(np.array(images) > top,255,np.multiply(images,255/top))
    return images
def extractForegroundMask(reverse: bool, mustExistInPreviousFrames: bool,static: bool,runImages, histLength, threshold,blur, pad: bool): #returns the filtered run images
    images = initializeImages(runImages)
    if blur > 0:
        for i in range(len(images)):
            images[i] = cv2.GaussianBlur(images[i],(blur,blur),cv2.BORDER_DEFAULT)
    fgbg = cv2.createBackgroundSubtractorMOG2(histLength,threshold,False)
    if reverse:
        for frameNumber in range(len(images)-1,-1,-1):
            if static and frameNumber < len(images)-histLength:
                images[frameNumber] = fgbg.apply(images[frameNumber],learningRate = 0)
            else:
                images[frameNumber] = fgbg.apply(images[frameNumber])
        if mustExistInPreviousFrames:
            for frameNumber in range(1,len(images)):
                images[frameNumber] = np.multiply(images[frameNumber],np.divide(images[frameNumber-1],255))
    else:
        for frameNumber in range(len(images)):
            if static and frameNumber > histLength:
                images[frameNumber] = fgbg.apply(images[frameNumber],learningRate = 0)
            else:
                images[frameNumber] = fgbg.apply(images[frameNumber])
        if mustExistInPreviousFrames:
            for frameNumber in range(len(images)-2,-1,-1):
                images[frameNumber] = np.multiply(images[frameNumber],np.divide(images[frameNumber+1],255))
    if pad:
        for frameNumber in range(len(images)):
            images[frameNumber] = np.pad(images[frameNumber],1,mode='constant',constant_values=255)
    return images
def overlayFrames(frame1,frame2): # returns the composite frame of two frames
    return np.multiply(frame1,np.divide(frame2,255))
def getRunsFromGroup(groupFolder): # str - name of grou p, str array - names of runs, str 2d array - timestamps in runs, str 2d array - filenames in runs
    filename_RFG = glob.glob(groupFolder + os.path.sep + '*.tif')
    groupName_RFG = os.path.basename(groupFolder)
    runNames_RFG = []
    runImages_RFG = []
    runTimestamps_RFG = []
    currRunName_RFG = ''
    for i in filename_RFG:
        tif_RFG = os.path.basename(i)
        runName_RFG, timestamp_RFG = tif_RFG.replace('.tif','').replace('.','').split('_')
        if runName_RFG != currRunName_RFG:
            currRunName_RFG = runName_RFG
            runNames_RFG.append(runName_RFG)
            runTimestamps_RFG.append([])
            runImages_RFG.append([])
        runTimestamps_RFG[-1].append(timestamp_RFG)
        runImages_RFG[-1].append(np.array(Image.open(i)))
    return groupName_RFG, runNames_RFG, runTimestamps_RFG, runImages_RFG
def concatFrames(image1,image2,axis):
    return np.concatenate((image1,image2),axis)
def imgNumStamps(frameNum, loc1, loc2, origGrid): # this def includes a frame stamp at [loc1][loc2] on origGrid with frameNum works with '-' character as well
    frameNums = []
    frameNumb = str(frameNum)
    for i in range(len(frameNumb)):
        if frameNumb[i] == '-':
            frameNums.append(-1)
        else:
            frameNums.append(frameNumb[i])
    numForms = ['111101101101111','010010010010010','111001111100111','111001111001111','101101111001001','111100111001111','111100111101111','111001001001001','111101111101111','111101111001111','000000111000000']
    grid = np.zeros((7,1+4*len(frameNums)),dtype=int)
    for i in range(len(frameNums)):
        for j in range(5):
            for k in range(3):
                grid[1+j][1+4*i+k] = numForms[int(frameNums[i])][3*j+k]    
    for i in range(7):
        for j in range(1+4*len(frameNums)):
            origGrid[loc1+i][loc2+j]=255*grid[i][j]
    return origGrid
def writeAviVideo(videoName, frameRate, allImages, openVideo: bool): # creates a video with videoname, frame_rate, using "images", and opens the video based on a boolean
    images = allImages
    images = np.array(images).astype(np.uint8)
    height, width = images[0].shape
    size = (width,height)
    out = cv2.VideoWriter(videoName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), frameRate, size, isColor=0) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()
    if openVideo:
        startfile(this_repo_path+os.path.sep+videoName+".avi")
def addLeadingZeros(finalLength, currText): # adds leading zeros to match the expected length of a number/string format
    currentText = str(currText)
    while len(currentText) < finalLength:
        currentText = '0' + currentText
    return currentText
groupNames = ['control 08 - 8 bit']#,'cs-137 05 - 8 bit','fiesta front w Be 10 - 8 bit'] # the short name of the folder containing images (tif files)
Images = [] # initializes the array used to store images to make a movie
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
detectedFrames = []
for groupName in groupNames:
    data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + groupName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    print(groupName)
    groupName, runNames, runTimesteps, thisGroupRunImages = getRunsFromGroup(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL
    allRunsInFolder = True
    if allRunsInFolder:
        runsOfInterest = range(len(runNames))
        batchName = 'All,'
    else:
        runsOfInterest = [15,16,17,19,25,34,38,41] # MUST be an array of run indices (0-indexed) #range(len(runNames)) to read all files in folder
        for i in range(len(runsOfInterest)):
            run = runsOfInterest.pop(0)
            if run <= len(runNames):
                runsOfInterest.append(run)
        batchName = ''
        for i in range(len(runsOfInterest)):
            batchName += str(runsOfInterest[i])+','
            runsOfInterest[i] -= 1
    answerKeyPath = glob.glob(data_folder_path+os.path.sep+'known*.txt')[0]
    answerKeyFile = open(answerKeyPath,'r')
    answerKeyLines = answerKeyFile.readlines()
    labels = []
    codeFrame = []
    keyFrame = []
    for runNumber in runsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
        thisRunImages = thisGroupRunImages[runNumber]
        thisRunImages.append(thisRunImages.pop(0))
        frames = []
        for frameNumber in range(len(thisRunImages)):
            frames.append(thisRunImages[frameNumber])
        thisRunImages = normalizePixelValues(thisRunImages)
        print(runNumber)
        # do stuff here
        hist = 50
        thresh = 5
        blur = 3
        thisRun1 = extractForegroundMask(False,True,True,thisRunImages, 50,9,0,False)
        # thisRun2 = extractForegroundMask(False,True,True,thisRunImages,hist,3,15,False)
        # thisRun3 = thisRunImages
        # thisRun3 = extractForegroundMask(False,False,True,thisRunImages,50,5,blur,False)
        thisRun4 = extractForegroundMask(False,True,True,thisRunImages,hist,100,35,False)
        ballParkFrame = 0
        detectedFrame = 0
        for frameNumber in range(len(thisRunImages)):
            if ballParkFrame == 0:
                if np.sum(thisRun4[frameNumber]) > 0:
                    ballParkFrame = frameNumber
        for frameNumber in range(len(thisRunImages)):  
            thisRun1[frameNumber] = overlayFrames(thisRun1[frameNumber],thisRun4[np.min([len(thisRunImages)-1,ballParkFrame+2])])
            if detectedFrame == 0 and np.sum(thisRun1[frameNumber]) > 0:
                detectedFrame = frameNumber
        detectedFrames.append(str(detectedFrame+1)+'-'+answerKeyLines[runNumber])
        codeFrame.append(detectedFrame+1)
        keyFrame.append(int(answerKeyLines[runNumber].split(' ')[0]))
        labels.append(str(runNumber+1))
        # thisRunImages = runFrameStamp(runNumber,thisRunImages)
        # thisImages = concatFrames(thisRunImages,thisRun1,2)
        thisImages = concatFrames(concatFrames(padRun(frames),padRun(thisRunImages),2),concatFrames(padRun(thisRun1),padRun(thisRun4),2),1)
        thisImages = runFrameStamp(runNumber,thisImages)
        low = detectedFrame
        high = int(answerKeyLines[runNumber].split(' ')[0])
        if high < low:
            low = high
            high = detectedFrame
        low = int(np.max([0,low-np.max([(high-low)/2,5])]))
        high = int(np.min([len(thisRunImages),high+np.max([(high-low)/2,5])]))
        for frameNumber in range(low,high):
            thisImages[frameNumber] = imgNumStamps(int(detectedFrame+1),10,0,thisImages[frameNumber])
            thisImages[frameNumber] = imgNumStamps(int(answerKeyLines[runNumber].split(' ')[0]),20,0,thisImages[frameNumber])
            Images.append(cv2.resize(thisImages[frameNumber],(2*115,89*2)))
    makeBarGraph(labels,0.35,['Code Output','Answer Key'],[codeFrame,keyFrame],groupName[0:-8],'Detection Frame','Event Number',False,True)
# writeAviVideo(videoName = 'Batch mini - misc.1',frameRate = 1,allImages = Images,openVideo = True)