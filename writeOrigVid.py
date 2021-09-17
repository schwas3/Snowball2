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

groupName = 'control 08 - 8 bit' # the short name of the folder containing images (tif files)

# WIP
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + groupName
data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
# the above line must include the path to the folder containing runs (i.e. 'PATH/control 08 - 8 bit')

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
    for i in range(len(images)):
        out.write(images[i])
    out.release()
    if openVideo:
        startfile(this_repo_path+os.path.sep+videoName+".avi")
def addLeadingZeros(finalLength, currentText): # adds leading zeros to match the expected length of a number/string format
    currentText = str(currentText)
    while len(currentText) < finalLength:
        currentText = '0' + currentText
    return currentText
groupName, runNames, runTimesteps, runImages = getRunsFromGroup(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL 

# --- These should be configured --- #
allRunsInFolder = True
if allRunsInFolder:
    runsOfInterest = range(len(runNames))
    batchName = 'Alll'
else:
    runsOfInterest = [36] # MUST be an array of run indices (0-indexed) #range(len(runNames)) to read all files in folder
    batchName = ''
    for i in range(len(runsOfInterest)):
        batchName += str(runsOfInterest[i])+','
        runsOfInterest[i] -= 1
writeVid=True # self explanatory
# --- #
images = [] # initializes the array used to store images to make a movie
batchName = ''
for i in range(len(runsOfInterest)):
    batchName += str(1+runsOfInterest[i]) + ','
correctedImages = [] # initializes the array used to store corrected images used for detection
for runNumber in runsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
    thisRunName = runNames[runNumber] # pulls the name of the run (i.e. the prefix)
    thisRunTimesteps = runTimesteps[runNumber] # pulls all the timesteps for the current run
    thisRunImages = runImages[runNumber] # pulls all the frames in the current run
    fgbg = cv2.createBackgroundSubtractorMOG2(history = 60,varThreshold = 24, detectShadows = False) # initializes the background subtractor MOG2
    thisRunCorrectedImages = []
    print(runNumber)
    for frameNumber in range(len(runImages[runNumber])): # iterates through every index in the range of the number of frames in the run
        thisFrameImage = thisRunImages[frameNumber] # gets the current frame
        thisFrameImage1=cv2.GaussianBlur(thisFrameImage,(45,45),cv2.BORDER_DEFAULT)
        thisFrameImage3=fgbg.apply(thisFrameImage1)
        thisFrameImage2=fgbg.apply(thisFrameImage)
        # thisFrameImage=fgbg.apply(thisFrameImage)
        # thisFrameImage3=cv2.GaussianBlur(thisFrameImage,(37,37),cv2.BORDER_DEFAULT)
        # thisFrameImage4=fgbg.apply(thisFrameImage2)
        # thisFrameCorrectedImage = fgbg.apply(thisFrameImage) # applies the background subtractor to the current frame
        # thisFrameCorrectedImage2 = fgbg.apply(thisFrameImage2) # applies the background subtractor to the current frame
        # thisRunCorrectedImages.append(thisFrameCorrectedImage)

        # --- completely asthetic video stuff starts here --- #
        if writeVid:
        #     cv2.rectangle(thisFrameImage, (0, 0), (30,7), 255,-1)
        #     cv2.putText(thisFrameImage, str(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,frameNumber)), (0, 6),
        #     cv2.FONT_HERSHEY_PLAIN, 0.5 , 0,bottomLeftOrigin=False)
            thisFrameImage = imgNumStamps(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,frameNumber),0,0,thisFrameImage)
        # thisFrameTrifoldImage = imgNumStamps(thisRunName,len(thisFrameTrifoldImage)-15,0,thisFrameTrifoldImage)
        # thisFrameTrifoldImage = imgNumStamps(addLeadingZeros(10,thisRunTimesteps[frameNumber]),len(thisFrameTrifoldImage)-8,0,thisFrameTrifoldImage)
            thisFrameComboImage1 = np.concatenate((thisFrameImage,thisFrameImage1),axis=1)
            thisFrameComboImage2 = np.concatenate((thisFrameImage2,thisFrameImage3),axis=1)
            thisFrameComboImage = np.concatenate((thisFrameComboImage1,thisFrameComboImage2),axis=0)
            # thisFrameComboImage2 = np.concatenate((thisFrameImage3,thisFrameImage4),axis=1)
            # thisFrameComboImage = np.concatenate((thisFrameComboImage1,thisFrameComboImage2),axis=0)
            # thisFrameComboImage = thisFrameComboImage1
            images.append(thisFrameComboImage)
        
        # thisRunTrifoldImages.append(thisFrameTrifoldImage)
        # above line is left in to allow for the creation of multiple separate videos separated by run
        # --- completely asthetic video stuff ends here --- #
if writeVid:
    writeAviVideo(videoName = groupName+' - '+batchName[0:-1],frameRate = 20,images = images,openVideo = True)