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
groupNames = ['fiesta front w Be 10 - 8 bit']#,'cs-137 05 - 8 bit','control 08 - 8 bit'] # the short name of the folder containing images (tif files)
threshBeta = 150
thresh = 10
histBig = 50
hist = 100
blurBig = 15
blur = 5
Images = [] # initializes the array used to store images to make a movie
writeVid=True # self explanatory
# WIP
for groupName in groupNames:
    this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
    this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
    github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
    ## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
    data_repo_name = "Snowball3"
    data_repo_path = github_path + os.path.sep + data_repo_name
    data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + groupName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    # the above line must include the path to the folder containing runs (i.e. 'PATH/control 08 - 8 bit')
    groupName, runNames, runTimesteps, runImages = getRunsFromGroup(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL 

    # --- These should be configured --- #
    allRunsInFolder = False
    if allRunsInFolder:
        runsOfInterest = range(len(runNames))
        batchName = 'Alll'
    else:
        runsOfInterest = [6,13,22,27,29,31,36,41,47] # MUST be an array of run indices (0-indexed) #range(len(runNames)) to read all files in folder
        batchName = ''
        for i in range(len(runsOfInterest)):
            batchName += str(runsOfInterest[i])+','
            runsOfInterest[i] -= 1
    # --- #
    batchName = ''
    correctedImages = [] # initializes the array used to store corrected images used for detection
    # blur = [1,3,5,7,9,11,13,15,17]
    # thresh = range(40,155,10)
    print(groupName)
    for runNumber in runsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
        thisRunName = runNames[runNumber] # pulls the name of the run (i.e. the prefix)
        thisRunTimesteps = runTimesteps[runNumber] # pulls all the timesteps for the current run
        thisRunImages = runImages[runNumber] # pulls all the frames in the current run
        threshBig = threshBeta#/255*(255-np.mean(thisRunImages))
        mid = int(np.mean(thisRunImages))
        bottom = 2*mid-255
        if bottom > 0:
            thisRunImages = np.where(np.subtract(thisRunImages,bottom)<0, 0,np.multiply(np.subtract(thisRunImages,bottom),255/(255-bottom)))
        fgbgForward = cv2.createBackgroundSubtractorMOG2(history = hist,varThreshold = thresh, detectShadows = False) # initializes the background subtractor MOG2
        fgbgReverse = cv2.createBackgroundSubtractorMOG2(history = hist,varThreshold = thresh, detectShadows = False) # initializes the background subtractor MOG2
        fgbgBig = cv2.createBackgroundSubtractorMOG2(history = histBig,varThreshold = threshBig, detectShadows = False) # initializes the background subtractor MOG2
        # fgbg = cv2.createBackgroundSubtractorMOG2(history = hist,varThreshold = thresh, detectShadows = False) # initializes the background subtractor MOG2
        # fgbgO = cv2.createBackgroundSubtractorMOG2(history = histO,varThreshold = threshO, detectShadows = False) # initializes the background subtractor MOG2
        thisRunCorrectedImages = []
        print(runNumber)
        originalFrameB = []
        # originalFrame = []
        reverseFrameB = []
        # reverseFrame = []
        blurredFrame = []
        bigFrame = []
        frame = []
        for i in range(len(thisRunImages)):
            bigFrame.append([])
            originalFrameB.append([])
            # originalFrame.append([])
            reverseFrameB.append([])
            # reverseFrame.append([])
            blurredFrame.append([])
            frame.append([])
        for frameNumber in range(1,len(thisRunImages)+1):
            thisFrameImage = thisRunImages[frameNumber%201]
            blurredFrame[frameNumber%201]=cv2.GaussianBlur(thisFrameImage,(blur,blur),cv2.BORDER_DEFAULT)
            # originalFrame[frameNumber%201]=fgbgO.apply(thisFrameImage)
            originalFrameB[frameNumber%201]=fgbgForward.apply(blurredFrame[frameNumber%201])
            if frameNumber < 51:
                bigFrame[frameNumber%201]=fgbgBig.apply(cv2.GaussianBlur(thisFrameImage,(blurBig,blurBig),cv2.BORDER_DEFAULT))
            else:
                bigFrame[frameNumber%201]=fgbgBig.apply(cv2.GaussianBlur(thisFrameImage,(blurBig,blurBig),cv2.BORDER_DEFAULT),learningRate=0)
            frame[frameNumber%201]=thisFrameImage
        for frameNumber in range(len(thisRunImages),0,-1):
            thisFrameImage = thisRunImages[frameNumber%201]
            reverseFrameB[frameNumber%201]=fgbgReverse.apply(blurredFrame[frameNumber%201])
        # topRow = np.concatenate((frame,blurredFrame),axis=1)
        # midRow = np.concatenate((originalFrame,originalFrameB),axis=1)
        # midRow2 = np.concatenate((reverseFrame,reverseFrameB),axis=1)
        compositeFrame = np.multiply(np.divide(originalFrameB,255),reverseFrameB)
        square = np.concatenate((frame,compositeFrame),axis=2)
        compositeFrame1 = np.multiply(np.divide(compositeFrame,255),bigFrame[0])
        # bigFrameHolder = []
        # for i in range(201):
        #     bigFrameHolder.append(bigFrame[0])
        square1 = np.concatenate((bigFrame,compositeFrame1),axis=2)
        square=np.concatenate((square,square1),axis=1)     
        # square = np.multiply(reverseFrameB,np.divide(originalFrameB,255))
        # square = np.where(np.multiply(reverseFrameB,originalFrameB)>0, 255, np.zeros_like(frame))
        # square = np.concatenate((topRow,midRow),axis=2)
        # square = np.concatenate((square,midRow2),axis=2)
        # square = np.concatenate((square,botRow),axis=2)
        if writeVid:
            for frameNumber in range(len(thisRunImages)):
                thisFrameImage = square[frameNumber]
                thisFrameImage = imgNumStamps(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,frameNumber%201),0,0,thisFrameImage)
                # thisFrameImage = imgNumStamps(thisRunName,len(thisFrameImage)-15,0,thisFrameImage)
                # thisFrameImage = imgNumStamps(addLeadingZeros(10,thisRunTimesteps[frameNumber%201]),len(thisFrameImage)-8,0,thisFrameImage)
                # if frameNumber == 0:
                Images.append(cv2.resize(thisFrameImage,(2*113,87*2)))
        # for frameNumber in range(0,-1*len(runImages[runNumber]),-1): # iterates through every index in the range of the number of frames in the run
        #     thisFrameImage = thisRunImages[frameNumber]
        #     thisFrameBlurredImage = cv2.GaussianBlur(thisFrameImage,(blur,blur),cv2.BORDER_DEFAULT)
        #     thisFrameCorrectedImage = fgbg.apply(thisFrameBlurredImage)
        #     # thisFrame = []
        #     # thisFrame.append([thisRunImages[frameNumber]]) # gets the current frame
        #     # for i in range(8):
        #     #     thisFrame.append(thisFrame[0])
        #     #     # thisFrame.append([cv2.GaussianBlur(thisFrame[0][0],(blur[i],blur[i]),cv2.BORDER_DEFAULT)])
        #     # for i in range(9):
        #     #     for j in range(8):
        #     #         thisFrame[i].append(fgbg[j].apply(thisFrame[i][0],learningRate = (j-1)/6))
        #     # thisFrameImage=fgbg.apply(thisFrameImage)
        #     # thisFrameImage3=cv2.GaussianBlur(thisFrameImage,(37,37),cv2.BORDER_DEFAULT)
        #     # thisFrameImage4=fgbg.apply(thisFrameImage2)
        #     # thisFrameCorrectedImage = fgbg.apply(thisFrameImage) # applies the background subtractor to the current frame
        #     # thisFrameCorrectedImage2 = fgbg.apply(thisFrameImage2) # applies the background subtractor to the current frame
        #     # thisRunCorrectedImages.append(thisFrameCorrectedImage)

        #     # --- completely asthetic video stuff starts here --- #
        #     if writeVid:
        #         image1 = np.concatenate((thisFrameImage,thisFrameBlurredImage),axis=1)
        #         image1 = np.concatenate((image1,thisFrameCorrectedImage),axis=1)
                
        #         # thisFrameTrifoldImage = np.concatenate((thisRunImages[0],thisRunImages[-1]),axis=1)
        #     #     cv2.rectangle(thisFrameImage, (0, 0), (30,7), 255,-1)
        #     #     cv2.putText(thisFrameImage, str(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,frameNumber)), (0, 6),
        #     #     cv2.FONT_HERSHEY_PLAIN, 0.5 , 0,bottomLeftOrigin=False)
        #         # thisFrameTrifoldImage = imgNumStamps(thisRunName,len(thisFrameTrifoldImage)-15,0,thisFrameTrifoldImage)
        #         # thisFrameTrifoldImage = imgNumStamps(addLeadingZeros(10,thisRunTimesteps[frameNumber]),len(thisFrameTrifoldImage)-8,0,thisFrameTrifoldImage)
        #         # for i in range(9):
        #         #     thisFrameRowImage = thisFrame[i][0]
        #         #     for j in range(1,9):
        #         #         thisFrameRowImage = np.concatenate((thisFrameRowImage,thisFrame[i][j]),axis=1)
        #         #     if i == 0:
        #         #         thisFrameComboImage = thisFrameRowImage
        #         #     else:
        #         #         thisFrameComboImage = np.concatenate((thisFrameComboImage,thisFrameRowImage),axis=0)
        #         image1 = imgNumStamps(addLeadingZeros(2,runNumber+1)+'-'+addLeadingZeros(3,200+frameNumber),0,0,image1)
        #         # thisFrameComboImage2 = np.concatenate((thisFrameImage3,thisFrameImage4),axis=1)
        #         # thisFrameComboImage = np.concatenate((thisFrameComboImage1,thisFrameComboImage2),axis=0)
        #         # thisFrameComboImage = thisFrameComboImage1
        #         images.insert(0,image1)
            
            # thisRunTrifoldImages.append(thisFrameTrifoldImage)
            # above line is left in to allow for the creation of multiple separate videos separated by run
            # --- completely asthetic video stuff ends here --- #
if writeVid:
    # writeAviVideo(videoName = groupName+' - '+batchName[0:-1],frameRate = 1,images = images,openVideo = True)
    writeAviVideo(videoName = 'Batch mini - misc.',frameRate = 1,images = Images,openVideo = True)