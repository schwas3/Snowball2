 #!~/.anaconda3/bin/python
import math
from PIL.Image import Image, new
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import cv2
import glob
from os import read, startfile, write
from os.path import exists
from numpy.core.numeric import allclose, zeros_like
from numpy.lib.function_base import rot90

def makeBarHistGraphsSolo(labels,width,barData,title,makeBar: bool,makeHist:bool,folderH):
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
        # suffix = 1
        # while exists(str(title)+' '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(folderH+str(title)+'.png')
    if makeHist:
        fig = plt.figure(figsize=(10,5.63))
        plt.title(title)
        plt.hist(barData,bins=8,alpha=0.5,rwidth=0.9)
        plt.xlabel('Detection Frame'+' (Mean = '+str(round(np.mean(barData),2))+', Std = '+str(round(np.std(barData),2))+')')
        plt.ylabel('Frequency')
        # suffix = 1
        # while exists(str(title)+' hist '+str(suffix)+'.png'):
        #     suffix += 1
        plt.savefig(folderH+str(title)+' hist.png')
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
    try:return np.pad(eventImages,[(0,0),(1,1),(1,1)],'constant',constant_values = 255)
    except:return np.pad(eventImages,[(0,0),(1,1),(1,1),(0,0)],'constant',constant_values = 255)
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
            images[frameNumber] = imgNumStamps(addLeadingZeros(2,eventNumber+1)+'-'+addLeadingZeros(3,(frameNumber)%len(images)),0,0,images[frameNumber])    
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
    filename_RFG = [os.path.join(runFolder,i) for i in os.listdir(runFolder)if i[-4:]=='.bmp' or i[-5:]=='.tiff'] # make sure is tiff and not .tif possible source of error
    # filename_RFG = [os.path.join(runFolder,i) for i in os.listdir(runFolder)] # make sure is tiff and not .tif possible source of error
    groupName_RFG = os.path.basename(runFolder)
    runNames_RFG = []
    runImages_RFG = []
    runTimestamps_RFG = []
    currRunName_RFG = ''
    valid_RFG = []
    for i in filename_RFG:
        tif_RFG = os.path.basename(i)
        try:
            runName_RFG, timestamp_RFG = tif_RFG.replace('.tiff','').replace('.tif','').split('_')
            if runName_RFG != currRunName_RFG:
                currRunName_RFG = runName_RFG
                runNames_RFG.append(runName_RFG)
                runTimestamps_RFG.append([])
                valid_RFG.append(True)
                runImages_RFG.append([])
            runImages_RFG[-1].append(cv2.imread(i))
            # if len(runImages_RFG)==35:
            #     runImages_RFG[-1].append(cv2.imread(i))
            if timestamp_RFG[-1] == 'X':
                valid_RFG[-1] = False
            runTimestamps_RFG[-1].append(timestamp_RFG.replace('X',''))
        except:
            pass
    return groupName_RFG, runNames_RFG, runTimestamps_RFG, runImages_RFG, valid_RFG
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
def writeAviVideo(videoName, frameRate, allImages, openVideo: bool,color: bool): # creates a video with videoname, frame_rate, using "images", and opens the video based on a boolean
    images = allImages
    images = np.array(images).astype(np.uint8)
    if not color:
        height, width = images[0].shape
        size = (width,height)
        out = cv2.VideoWriter(videoName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), frameRate, size, isColor=0) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
    else:
        height, width,layers = images[0].shape
        size = (width,height)
        out = cv2.VideoWriter(videoName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), frameRate, size) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
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
data_repo_name = "Snowball7"
data_repo_name = "Snowball8"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES'
data_folder_name = 'ColorCroppedTiffs'
folder = 'run05'
runNames = glob.glob(data_repo_path + os.path.sep +data_folder_name + os.path.sep + folder + os.path.sep + '*')
for i in range(len(runNames)):
    runNames[i] = os.path.basename(runNames[i])
# print(runNames)
runNames = ['Cs-137'] # the short name of the folder containing images (tif files)
notesContent = []
try: 
    os.mkdir(this_repo_path+os.path.sep+folder) 
except:
    pass
for runName in runNames:
    Images3 = []
    Images4 = []
    runEventsNoteContent = runName+', Invalid Events: '
    invalidEventsInRun = ''
    detectedFrames = []
    data_folder_name = 'SNOWBALL CROPPED IMAGES'
    data_folder_name = 'ColorCroppedTiffs'
    data_folder_name += os.path.sep + folder
    data_folder_name += os.path.sep + runName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    runName, eventPrefixes, eventFrameTimestamps, runEventImages, validRunEvents = getEventsFromRun(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL
    print(str(runName)+'/'+str(len(eventPrefixes)))
    allEventsInFolder = False
    if allEventsInFolder:
        eventsOfInterest = np.arange(len(eventPrefixes))
    else:
        eventsOfInterest = np.array([8]) # 1-indexing
        eventsOfInterest = np.array([1,2,3,4,5,6,8,15,35]) # 1-indexing
        eventsOfInterest = eventsOfInterest[eventsOfInterest <= len(eventPrefixes)]
        for i in range(len(eventsOfInterest)):
            eventsOfInterest[i] -= 1
    numEvents = len(eventsOfInterest)
    # to-do: remove following lines
    # answerKeyPath = glob.glob(data_folder_path+os.path.sep+'known*.txt')[0]
    # answerKeyFile = open(answerKeyPath,'r')
    # answerKeyLines = answerKeyFile.readlines()
    labels = []
    codeFrame = []
    keyFrame = []
    for eventNumber in eventsOfInterest: # iterates over all runNumber in runsOfInterest (note: runNumber is 0-indexed)
        eventLabel = str(eventNumber+1)
        print(eventNumber)
        if not validRunEvents[eventNumber]:
            runEventsNoteContent+= eventLabel+','
            eventLabel = '('+eventLabel+')'
        thisEventImages = runEventImages[eventNumber]
        thisEventFrameTimestamps = eventFrameTimestamps[eventNumber]
        eventLength = len(thisEventImages)
        if thisEventFrameTimestamps[0][0] == '0':
            thisEventImages.append(thisEventImages.pop(0)) # the 0-th frame is removed and added to the end of the event images
            thisEventFrameTimestamps.append(thisEventFrameTimestamps.pop(0)) # the 0-th frame is removed and added to the end of the event images
        # thisEventImages = cv2.normalize(np.array(thisEventImages),np.zeros_like(thisEventImages),0,255,cv2.NORM_MINMAX) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        thisEventImagesG = extractForegroundMask(False,True,True,thisEventImages,50,100,35,0)
        thisEventImagesGGG = extractForegroundMask(False,True,True,thisEventImages,50,100,1,0)
        for frameNumber in range(len(thisEventImages)):
            thisEventImagesG[frameNumber] = overlayFrames(thisEventImagesGGG[frameNumber],thisEventImagesG[min([len(thisEventImages)-1,frameNumber+2])])
            # thisEventImagesG[frameNumber] = overlayFrames(thisEventImagesGGG[frameNumber],thisEventImagesG[-1])
        # thisEventImagesG = [overlayFrames(thisEventImagesGs,thisEventImagesG[-1])for thisEventImagesGs in thisEventImagesG]
        blur = 1
        allCircles = np.zeros_like(thisEventImages)
        thisEventImagesG = [np.subtract(255,thisEventImagesGs) for thisEventImagesGs in thisEventImagesG]
        thisEventImagesG = [cv2.GaussianBlur(thisEventImagesGs,(blur,blur),cv2.BORDER_DEFAULT)for thisEventImagesGs in thisEventImagesG]
        thisEventImagesG = np.array(thisEventImagesG).astype(np.uint8)
        # thisEventImagesG = [np.subtract(255,cv2.cvtColor(thisEventImage,cv2.COLOR_RGB2GRAY))for thisEventImage in thisEventImages]
        # thisEventImages = np.array(thisEventImages)
        # thisEvent3 = np.array(extractForegroundMask(False,True,True,thisEventImages,50,100,35,0)).astype(np.uint8)
        # thisEvent3 = np.array(np.subtract(255,thisEvent3)).astype(np.uint8)
        # thisEvent1 = np.array(extractForegroundMask(False,False,True,thisEventImages,50,9,3,0)).astype(np.uint8)
        # thisEvent1 = extractForegroundMask(False,False,True,thisEventImages,50,9,0,0)
        # thisEvent1 = np.array(np.subtract(255,thisEvent1)).astype(np.uint8)
        # newImage = [cv2.cvtColor(thisEvent,cv2.COLOR_GRAY2RGB) for thisEvent in thisEvent3]
        # newImage1 = [cv2.cvtColor(thisEvent,cv2.COLOR_GRAY2RGB) for thisEvent in thisEvent1]
        # newImage = np.array(newImage).astype(np.uint)
        foundIt = False
        thisEventImagesGg = []
        for frameNumber in range(len(thisEventImages)):
            # thisEvent1[frameNumber] = np.subtract(255,overlayFrames(thisEvent1[frameNumber],thisEvent3[min([frameNumber+12,len(thisEventImages)-1])]))
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 220
            params.maxThreshold = 256
            # params.thresholdStep = 255
            params.filterByArea = False
            params.minArea = 0
            # params.maxArea = 
            params.filterByCircularity = False
            params.minCircularity = 0
            params.filterByConvexity = False
            params.minConvexity = 0
            params.filterByInertia = False
            params.minInertiaRatio = 0
            detector = cv2.SimpleBlobDetector_create()
            keypoints = detector.detect(thisEventImagesG[frameNumber])
            thisEventImages[frameNumber] = cv2.drawKeypoints(thisEventImages[frameNumber],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if frameNumber > 0:
                allCircles[frameNumber] = cv2.drawKeypoints(allCircles[frameNumber-1],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            thisEventImagesGg.append(cv2.merge([thisEventImagesG[frameNumber],thisEventImagesG[frameNumber],thisEventImagesG[frameNumber]]))
            thisEventImagesGg[frameNumber] = cv2.drawKeypoints(thisEventImagesGg[frameNumber],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # print(eventNumber,frameNumber)
            # [print(keypoint.pt,keypoint.size)for keypoint in keypoints]
        
            # detector = cv2.SimpleBlobDetector_create(params)
            # keypoints = detector.detect(thisEvent3[frameNumber])
            # if len(keypoints) > 0:
            #     if not foundIt:
            #         [Images.append(concatFrames(concatFrames(thisEventImages[frameNumber-i],thisEventImagesGg[frameNumber-i],1),)for i in range(3,0,-1)]
            #     foundIt = True
            # thisEventImages[frameNumber] = cv2.drawKeypoints(thisEventImages[frameNumber],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            # newImage[frameNumber] = cv2.drawKeypoints(newImage[frameNumber],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # newImage1[frameNumber] = cv2.drawKeypoints(thisEvent1[frameNumber],keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            # # circles = np.array([])
            # circles = cv2.HoughCircles(thisEvent3[frameNumber],cv2.HOUGH_GRADIENT,1,10,np.array([]),10,10,1,30)
            # if len(circles) > 0:
            #     print(circles)
            #     circles = np.round(circles[0, :]).astype("int")
            #     print(circles)
            #     for (x,y,r) in circles:
            #         newImage[frameNumber] = cv2.circle(newImage[frameNumber],(x,y),1,(0,100,100))
            #         newImage[frameNumber] = cv2.circle(newImage[frameNumber],(x,y),r,(255,0,255))
                # Images.append(newImage[frameNumber])
            # if foundIt:
                # Images.append(thisEventImages[frameNumber])
            Images.append(concatFrames(concatFrames(thisEventImages[frameNumber],thisEventImagesGg[frameNumber],1),allCircles[frameNumber],1))
                # Images.append(concatFrames(thisEventImages[frameNumber],newImage[frameNumber],1))
                # Images.append(concatFrames(thisEventImages[frameNumber],concatFrames(newImage[frameNumber],newImage1[frameNumber],1),1))
writeAviVideo(videoName = folder+os.path.sep+'Blob Detection - '+folder,frameRate = 1,allImages = Images,openVideo = True,color = True)