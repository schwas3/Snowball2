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

def makeBarHistGraphsSolo(labels,width,barData,title,makeBar: bool,makeHist:bool):
    if makeBar:
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10,5.63))
        rects = ax.bar(x,barData,width,label = 'Code Output ( (Mean = '+str(round(np.mean(barData),2))+', Std = '+str(round(np.std(barData),2))+')')
        ax.set_ylabel('Detection Frame',fontsize = 10)
        ax.set_xlabel('Event Number',fontsize = 10)
        ax.set_title(title,fontsize = 12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,fontsize = 6)
        ax.legend(fontsize = 8)
        ax.bar_label(rects, padding=2, fontsize = 6)
        fig.tight_layout()
        suffix = 1
        # while exists(str(title)+' '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(str(title)+' '+str(suffix)+'.png')
    if makeHist:
        fig = plt.figure(figsize=(10,5.63))
        plt.title(title)
        plt.hist(barData,bins=8,alpha=0.5,rwidth=0.9)
        plt.xlabel('Detection Frame'+' (Mean = '+str(round(np.mean(barData),2))+', Std = '+str(round(np.std(barData),2))+')')
        plt.ylabel('Frequency')
        suffix = 1
        # while exists(str(title)+' hist '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(str(title)+' hist '+str(suffix)+'.png')
def makeBarGraph(labels,width,barLabels,barData,title,yLabel,xLabel, makeBar: bool, makeHist: bool):
    if makeBar:
        x = np.arange(len(labels))  # the label locations
        fig, ax = plt.subplots(figsize=(10,5.63))
        rects1 = ax.bar(x - width/2, barData[0], width, label = barLabels[0]+' (Mean = '+str(round(np.mean(barData[0]),2))+', Std = '+str(round(np.std(barData[0]),2))+')')
        rects2 = ax.bar(x + width/2, barData[1], width, label = barLabels[1]+' (Mean = '+str(round(np.mean(barData[1]),2))+', Std = '+str(round(np.std(barData[1]),2))+')')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(yLabel,fontsize = 10)
        ax.set_xlabel(xLabel,fontsize = 10)
        ax.set_title(title,fontsize = 12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,fontsize = 6)
        ax.legend(fontsize = 8)

        ax.bar_label(rects1, padding=2, fontsize = 6)
        ax.bar_label(rects2, padding=2,fontsize = 6)

        fig.tight_layout()
        suffix = 0
        # while exists(str(title)+' '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(str(title)+' '+str(suffix)+'.png')
    if makeHist:
        suffix = 0
        fig = plt.figure(figsize=(10,5.63))
        plt.subplot(1,2,1,title='Code Output' + ' ('+str(title)+')')
        plt.hist(barData[0],bins=8,alpha=0.5,rwidth=0.9)
        plt.xlabel('Detection Frame'+' (Mean = '+str(round(np.mean(barData[0]),2))+', Std = '+str(round(np.std(barData[0]),2))+')')
        plt.ylabel('Frequency')
        plt.subplot(1,2,2,title='Answer Key' + ' ('+str(title)+')')
        plt.xlabel('Detection Frame'+' (Mean = '+str(round(np.mean(barData[1]),2))+', Std = '+str(round(np.std(barData[1]),2))+')')
        plt.ylabel('Frequency')
        plt.hist(barData[1],bins=8,alpha=0.5,rwidth=0.9)
        # while exists(str(title)+' hist '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(str(title)+' hist '+str(suffix)+'.png')
def padEvent(eventImages):
    return np.pad(eventImages,[(0,0),(1,1),(1,1)],'constant',constant_values = 255)
def initializeImages(eventImages):
    newImages = []
    for eventImage in eventImages:
        newImages.append(eventImage)
    return newImages
def eventFrameStamp(eventNumber,eventImages,eventPrefix,eventTimestamps,labels: bool):
    images = initializeImages(eventImages)
    if labels:
        for frameNumber in range(len(images)):
            images[frameNumber] = imgNumStamps(addLeadingZeros(2,eventNumber+1)+'-'+addLeadingZeros(3,(frameNumber)%len(images)),0,0,images[frameNumber])
            images[frameNumber] = imgNumStamps(eventPrefix,len(images[frameNumber])-13,0,images[frameNumber])
            images[frameNumber] = imgNumStamps(addLeadingZeros(10,eventTimestamps[frameNumber]),len(images[0])-7,0,images[frameNumber])
    else:
        for frameNumber in range(len(images)):
            images[frameNumber] = imgNumStamps(addLeadingZeros(2,eventNumber+1)+'-'+addLeadingZeros(3,(eventNumber+1)%len(images)),0,0,images[frameNumber])    
    return images
def normalizePixelValues(eventImages,lowerLimit,upperLimit):
    images = np.array(initializeImages(eventImages))
    holder = np.zeros_like(images)
    mid = int(np.mean(images))
    if mid > 255/2:
        bottom = np.min([lowerLimit,2*mid-255])
        images = cv2.normalize(images,holder,bottom*255/(bottom-255),255,cv2.NORM_MINMAX)
    else:
        top = np.max([2*mid,upperLimit])
        images = cv2.normalize(images,holder,0,255*255/top,cv2.NORM_MINMAX)
    return images
def extractForegroundMask(reverse: bool, mustExistInPreviousFrames: bool,static: bool,eventImages, histLength, threshold,blur,startingFrom): #returns the filtered run images
    images = initializeImages(eventImages)
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
            if startingFrom == 0 or startingFrom > len(images)-2:
                startingFrom = len(images)-2
            for frameNumber in range(startingFrom,-1,-1):
                images[frameNumber] = np.multiply(images[frameNumber],np.divide(images[frameNumber+1],255))
    return images
def overlayFrames(frame1,frame2): # returns the composite frame of two frames
    return np.multiply(frame1,np.divide(frame2,255))
def getEventsFromRun(runFolder): # str - name of grou p, str array - names of runs, str 2d array - timestamps in runs, str 2d array - filenames in runs
    filename_RFG = glob.glob(runFolder + os.path.sep + '*.tiff')
    groupName_RFG = os.path.basename(runFolder)
    runNames_RFG = []
    runImages_RFG = []
    runTimestamps_RFG = []
    currRunName_RFG = ''
    for i in filename_RFG:
        tif_RFG = os.path.basename(i)
        runName_RFG, timestamp_RFG = tif_RFG.replace('.tiff','').replace('.tif','').split('_')
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
Images = [] # initializes the array used to store images to make a movie
Images1 = [] # initializes the array used to store images to make a movie
Images2 = [] # initializes the array used to store images to make a movie
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball5"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES'
folder = 'e'
runNames = glob.glob(data_repo_path + os.path.sep +data_folder_name + os.path.sep + folder + os.path.sep + '*')
for i in range(len(runNames)):
    runNames[i] = os.path.basename(runNames[i])
# print(runNames)
# runNames = ['control0'] # the short name of the folder containing images (tif files)
for runName in runNames:
    detectedFrames = []
    data_folder_name = 'SNOWBALL CROPPED IMAGES'
    data_folder_name += os.path.sep + folder
    data_folder_name += os.path.sep + runName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    runName, eventPrefixes, eventFrameTimestamps, runEventImages = getEventsFromRun(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL
    print(str(runName)+'/'+str(len(eventPrefixes)))
    allEventsInFolder = True
    if allEventsInFolder:
        eventsOfInterest = np.arange(len(eventPrefixes))
    else:
        eventsOfInterest = np.array([12,15,18,21]) # 1-indexing
        eventsOfInterest = eventsOfInterest[eventsOfInterest <= len(eventPrefixes)]
        for i in range(len(eventsOfInterest)):
            eventsOfInterest[i] -= 1
    # to-do: remove following lines
    # answerKeyPath = glob.glob(data_folder_path+os.path.sep+'known*.txt')[0]
    # answerKeyFile = open(answerKeyPath,'r')
    # answerKeyLines = answerKeyFile.readlines()
    labels = []
    codeFrame = []
    keyFrame = []
    for eventNumber in eventsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
        thisEventImages = runEventImages[eventNumber]
        thisEventFrameTimestamps = eventFrameTimestamps[eventNumber]
        eventLength = len(thisEventImages)
        if thisEventFrameTimestamps[0][0] == '0':
            thisEventImages.append(thisEventImages.pop(0)) # the 0-th frame is removed and added to the end of the event images
            thisEventFrameTimestamps.append(thisEventFrameTimestamps.pop(0)) # the 0-th frame is removed and added to the end of the event images
        frames = []
        for frameNumber in range(eventLength):
            frames.append(thisEventImages[frameNumber])
        thisEventImages = normalizePixelValues(thisEventImages,30,225) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        print(eventNumber)
        # do stuff here
        # thisEvent1 = extractForegroundMask(False,True,True,thisEventImages, 50,9,0,0)
        thisEvent3 = extractForegroundMask(False,True,True,thisEventImages,50,100,35,0)
        ballParkFrame = 0
        detectedFrame = 0
        for frameNumber in range(eventLength):
            if ballParkFrame == 0:
                if np.sum(thisEvent3[frameNumber]) > 0:
                    ballParkFrame = frameNumber
        thisEvent1 = extractForegroundMask(False,True,True,thisEventImages,ballParkFrame - 10,9,0,ballParkFrame+2) # changed hist from - 10 to - 5 and ballparkframe + 2 from + 5
        thisEvent2 = extractForegroundMask(False,True,True,thisEventImages,ballParkFrame - 10,100,35,ballParkFrame+2)
        ballParkFrame = 0
        for frameNumber in range(eventLength):
            if ballParkFrame == 0:
                if np.sum(thisEvent2[frameNumber]) > 0:
                    ballParkFrame = frameNumber 
        for frameNumber in range(eventLength):
            thisEvent1[frameNumber] = overlayFrames(thisEvent1[frameNumber],thisEvent2[np.min([len(thisEventImages)-1,ballParkFrame+2])])
            if detectedFrame == 0 and np.sum(thisEvent1[frameNumber]) > 0:
                detectedFrame = frameNumber
        if detectedFrame == 0:
            detectedFrame = eventLength
        detectedFrames.append(str(detectedFrame)+','+eventPrefixes[eventNumber]+'_'+thisEventFrameTimestamps[(detectedFrame)%eventLength]+'\n')
        # detectedFrames.append(str(detectedFrame)) #+'-'+answerKeyLines[eventNumber])
        codeFrame.append(detectedFrame)
        # keyFrame.append(int(answerKeyLines[eventNumber].split(' ')[0]))
        # labels.append(str(eventNumber+1))
        theseImages = concatFrames(padEvent(frames),padEvent(thisEvent1),2)
        tStamp = []
        for timestamp in thisEventFrameTimestamps:
            tStamp.append(timestamp.replace('.',''))
        # # below is purely comparison based
        low = np.max([0,detectedFrame - 5])
        high = np.min([eventLength,detectedFrame + 5])
        # low = 0
        # high = eventLength
        # low = detectedFrame
        # high = int(answerKeyLines[eventNumber].split(' ')[0])
        # if high < low:
        #     low = high
        #     high = detectedFrame
        # low = int(np.max([0,low-np.max([(high-low)/2,5])]))
        # high = int(np.min([eventLength,high+np.max([(high-low)/2,5])]))
        thisImages = []
        for frameNumber in range(eventLength):
            thisImages.append(cv2.resize(theseImages[frameNumber],(256,96)))
            thisImages[frameNumber] = imgNumStamps(int(detectedFrame),7,0,thisImages[frameNumber])
        thisImages = eventFrameStamp(eventNumber,thisImages,eventPrefixes[eventNumber].replace('.',''),tStamp,True)
        for frameNumber in range(eventLength):
            # thisImages[frameNumber] = imgNumStamps(int(answerKeyLines[eventNumber].split(' ')[0]),20,0,thisImages[frameNumber])
        # leave stamp code output? (seems very useful)
            # if runName == 'control2' or runName == 'control3' or runName == 'old':
            #     Images1.append(cv2.resize(thisImages[frameNumber],(256,96)))
            # else:
            Images.append(thisImages[frameNumber])
            if frameNumber <= high and frameNumber >= low:
                Images2.append(thisImages[frameNumber])
    makeBarHistGraphsSolo(labels,0.35,codeFrame,runName,False,False)
    if False:
        txtName = runName
        txtFile = open(this_repo_path+os.path.sep+txtName+' - Results','w')
        fileContents = "".join(detectedFrames)
        txtFile.write(fileContents)
        txtFile.close()
writeAviVideo(videoName = 'Full Runs - '+folder,frameRate = 1,allImages = Images,openVideo = True)
# writeAviVideo(videoName = folder+' - Full Runs 2 of 2',frameRate = 1,allImages = Images1,openVideo = False)
writeAviVideo(videoName ='Detection Clips - '+folder,frameRate = 1,allImages = Images2,openVideo = False)