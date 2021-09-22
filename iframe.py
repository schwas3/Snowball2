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
from os import read, startfile
from os.path import exists
import csv
blur = 5
blurBig = 15
thresh = 10
threshBig = 150
hist = 100
histBig = 50
detectedFrames = []
groupNames = ['control 08 - 8 bit','cs-137 05 - 8 bit','fiesta front w Be 10 - 8 bit']#['control 06 - 8 bit','control 07 - 8 bit','control 10 - 8 bit','AmBe 01 - 8 bit','cs137 06 - 8 bit'] # the short name of the folder containing images (tif files)
txtName = ''
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
def getRunsFromGroup(groupFolder): # str - name of group, str array - names of runs, str 2d array - timestamps in runs, str 2d array - filenames in runs
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
def imgNumStamps(frameNum, loc1, loc2, origGrid): # this def includes a frame stamp at [loc1][loc2] on origGrid with frameNum works with '-' character as well
    frameNums = []
    frameNum = str(frameNum)
    for i in range(len(frameNum)):
        if frameNum[i] == '-':
            frameNums.append(-1)
        else:
            frameNums.append(frameNum[i])
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
def writeAviVideo(videoName, frameRate, images, openVideo: bool): # creates a video with videoname, frame_rate, using "images", and opens the video based on a boolean
    images = np.array(images).astype(np.uint8)
    height, width = images[0].shape
    size = (width,height)
    out = cv2.VideoWriter(videoName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), frameRate, size, isColor=0) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
    for img in images:
        out.write(img)
    out.release()
    if openVideo:
        startfile(this_repo_path+os.path.sep+videoName+".avi")
def addLeadingZeros(finalLength, currentText): # adds leading zeros to match the expected length of a number/string format
    currentText = str(currentText)
    while len(currentText) < finalLength:
        currentText = '0' + currentText
    return currentText
for groupName in groupNames:
    ## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
    data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + groupName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    # the above line must include the path to the folder containing runs (i.e. 'PATH/control 08 - 8 bit')
    groupName, runNames, runTimesteps, runImages = getRunsFromGroup(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL 
    detectedFrames.append(groupName+'\n')
    allRunsInFolder = True
    if allRunsInFolder:
        runsOfInterest = range(len(runNames))
        batchName = 'Alll'
    else:
        runsOfInterest = [34] # MUST be an array of run indices (0-indexed) #range(len(runNames)) to read all files in folder
        batchName = ''
        for i in range(len(runsOfInterest)):
            batchName += str(runsOfInterest[i])+','
            runsOfInterest[i] -= 1
    answerKeyPath = glob.glob(data_folder_path+os.path.sep+'known*.txt')[0]
    answerKeyFile = open(answerKeyPath,'r')
    answerKeyLines = answerKeyFile.readlines()
    correctedImages = [] # initializes the array used to store corrected images used for detection
    print(groupName)
    for runNumber in runsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
        print(runNumber)
        thisRunName = runNames[runNumber] # pulls the name of the run (i.e. the prefix)
        thisRunTimesteps = runTimesteps[runNumber] # pulls all the timesteps for the current run
        thisRunImages = runImages[runNumber] # pulls all the frames in the current run
        runLength = len(thisRunImages)
        mid = int(np.mean(thisRunImages))
        bottom = 2*mid-255
        if bottom > 0:
            thisRunImages = np.where(np.array(thisRunImages) < bottom, 0,np.multiply(np.subtract(thisRunImages,bottom),255/(255-bottom)))
        else:
            top = 2*mid
            thisRunImages = np.where(np.array(thisRunImages) > top,255,np.multiply(thisRunImages,255/top))
        fgbgForward = cv2.createBackgroundSubtractorMOG2(history = hist,varThreshold = thresh, detectShadows = False) # initializes the background subtractor MOG2
        fgbgReverse = cv2.createBackgroundSubtractorMOG2(history = hist,varThreshold = thresh, detectShadows = False) # initializes the background subtractor MOG2
        fgbgBig = cv2.createBackgroundSubtractorMOG2(history = histBig,varThreshold = threshBig, detectShadows = False) # initializes the background subtractor MOG2
        detectedFrame = 0
        blurredImages = []
        forwardImages = []
        reverseImages = []
        bigImages = []
        bigFrameVal = []
        frameVal = []
        compositeImages = []
        for i in range(runLength):
            blurredImages.append([])
            forwardImages.append([])
            reverseImages.append([])
            bigImages.append([])
            frameVal.append(0)
            bigFrameVal.append(0)
            compositeImages.append([])
        for frameNumber in range(1,runLength+1): # iterates through every index in the range of the number of frames in the run
            thisFrameImage = thisRunImages[frameNumber%runLength] # gets the current frame
            blurredImages[frameNumber%runLength]=cv2.GaussianBlur(thisFrameImage,(blur,blur),cv2.BORDER_DEFAULT)
            forwardImages[frameNumber%runLength]=fgbgForward.apply(blurredImages[frameNumber%runLength]) # applies the background subtractor to the current frame in forward
            if frameNumber < 51:
                bigImages[frameNumber%runLength]=fgbgBig.apply(cv2.GaussianBlur(thisFrameImage,(blurBig,blurBig),cv2.BORDER_DEFAULT))
            else:
                bigImages[frameNumber%runLength]=fgbgBig.apply(cv2.GaussianBlur(thisFrameImage,(blurBig,blurBig),cv2.BORDER_DEFAULT),learningRate=0)
            bigFrameVal[frameNumber%runLength]=np.sum(bigImages[frameNumber%runLength])
        for frameNumber in range(runLength,0,-1): # iterates in reverse
            reverseImages[frameNumber%runLength]=fgbgReverse.apply(np.array(blurredImages[frameNumber%runLength])) # applies the background subtractor to the current frame in reverse
        compositeImages = np.multiply(np.divide(forwardImages,255),reverseImages)
        ballParkFrame = 0
        for frameNumber in range(50,runLength):
            tempFrameNumber = frameNumber
            frameIsValid = ballParkFrame == 0
            for i in range(5):
                if frameIsValid and tempFrameNumber <= runLength+1 and bigFrameVal[tempFrameNumber%runLength] == 0:
                    frameIsValid = False
                tempFrameNumber += 1
            if frameIsValid:
                ballParkFrameTest = bigImages[frameNumber]
                for offFrameNumber in range(np.min([frameNumber+1,runLength]),np.min([frameNumber+8,runLength+1])):
                    ballParkFrameTest = np.multiply(ballParkFrameTest,np.divide(bigImages[offFrameNumber%runLength],255))
                    if np.sum(ballParkFrameTest) == 0:
                        frameIsValid = False
                if frameIsValid:
                    ballParkFrame = frameNumber
        for frameNumber in range(runLength):
            compositeImages[frameNumber] = np.multiply(compositeImages[frameNumber],np.multiply(np.divide(bigImages[np.min([frameNumber+3,runLength])%runLength],255),np.divide(bigImages[np.min([frameNumber+6,runLength])%runLength],255)))
            frameVal[frameNumber]=np.sum(compositeImages[frameNumber])
        for frameNumber in range(ballParkFrame-30,runLength):
            if detectedFrame == 0:
                detectedFrame = frameNumber
                for offFrameNumber in range(frameNumber,np.min([frameNumber+3,runLength+1])):
                    if frameVal[offFrameNumber%runLength] == 0:
                        detectedFrame = 0
        detectedFrames.append(str(detectedFrame)+' - '+answerKeyLines[runNumber])
    # txtName += groupName+','
txtName = 'Batch - Results' + ' - hist='+str(hist)+',vT='+str(thresh)+',blur='+str(blur)+','
txtPath = this_repo_path+os.path.sep+txtName
suffix = 0
while exists(txtPath+str(suffix)):
    suffix += 1
txtFile = open(txtPath+str(suffix),'w')
fileContents = "".join(detectedFrames)
txtFile.write(fileContents)
txtFile.close()