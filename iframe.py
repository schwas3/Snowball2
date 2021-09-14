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
from os import startfile

# WIP
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + 'control 08 - 8 bit'
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
def getBackground(images, startFrame, stopFrame): # returns a background by averaging the images between startFrame and stopFrame from provided images
    background = np.zeros([len(images[0]),len(images[0][0])],dtype=int)
    for i in range(startFrame,stopFrame):
        background += images[i]
    background = background/(stopFrame - startFrame)
    return background
def backgroundCorrected(background, images, scale: bool): # subtracts background from images, if scale all images are shifted so that 0-max(background) => 0 while 255 => 255
    images = images - background
    if scale:
        background_max = np.max(background)
        images += background_max
        images *= 255/(255+background_max)
    images = np.where(images < 0, 0, images)
    return images
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
def writeAviVideo(videoname, frame_rate, images, openVideo: bool): # creates a video with videoname, frame_rate, using "images", and opens the video based on a boolean
    images = np.array(images).astype(np.uint8)
    height, width = images[0].shape
    size = (width,height)
    out = cv2.VideoWriter(videoname+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size, isColor=0) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()
    if openVideo:
        startfile(this_repo_path+os.path.sep+videoname+".avi")
def addLeadingZeros(finalLength, currentText): # adds leading zeros to match the expected length of a number/string format
    currentText = str(currentText)
    while len(currentText) < finalLength:
        currentText = '0' + currentText
    return currentText
groupName, runNames, runTimesteps, runImages = getRunsFromGroup(data_folder_path)
trifoldImages = []
for runNumber in range(len(runNames)):
    thisRunName = runNames[runNumber]
    thisRunTimesteps = runTimesteps[runNumber]
    thisRunImages = runImages[runNumber]
    thisRunBackground = getBackground(thisRunImages,1,51)
    thisRunCorrectedImages = backgroundCorrected(thisRunBackground,thisRunImages,scale = False)
    thisRunCorrectedScaledImages = backgroundCorrected(thisRunBackground,thisRunImages,scale = True)
    thisRunTrifoldImages = []
    for frameNumber in range(len(runImages[runNumber])):
        thisFrameTrifoldImage = np.concatenate((thisRunImages[frameNumber],thisRunCorrectedImages[frameNumber],thisRunCorrectedScaledImages[frameNumber]),axis=1)
        thisFrameTrifoldImage = imgNumStamps(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,frameNumber),0,0,thisFrameTrifoldImage)
        thisFrameTrifoldImage = imgNumStamps(thisRunName,len(thisFrameTrifoldImage)-15,0,thisFrameTrifoldImage)
        thisFrameTrifoldImage = imgNumStamps(addLeadingZeros(10,thisRunTimesteps[frameNumber]),len(thisFrameTrifoldImage)-8,0,thisFrameTrifoldImage)
        thisRunTrifoldImages.append(thisFrameTrifoldImage)
        trifoldImages.append(thisFrameTrifoldImage)
writeVid=True
if writeVid:
    writeAviVideo('control 08 - 8 bit',60,trifoldImages[0:],True)