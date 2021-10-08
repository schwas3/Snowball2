 #!~/.anaconda3/bin/python
import math
from PIL.Image import new
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import cv2
import glob
from os import read, startfile, write
from os.path import exists
from numpy.core.numeric import zeros_like
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
    filename_RFG = glob.glob(runFolder + os.path.sep + '*.tiff') # make sure is tiff and not .tif possible source of error
    groupName_RFG = os.path.basename(runFolder)
    runNames_RFG = []
    runImages_RFG = []
    runTimestamps_RFG = []
    currRunName_RFG = ''
    valid_RFG = []
    for i in filename_RFG:
        tif_RFG = os.path.basename(i)
        runName_RFG, timestamp_RFG = tif_RFG.replace('.tiff','').replace('.tif','').split('_')
        if runName_RFG != currRunName_RFG:
            currRunName_RFG = runName_RFG
            runNames_RFG.append(runName_RFG)
            runTimestamps_RFG.append([])
            valid_RFG.append(True)
            runImages_RFG.append([])
        runImages_RFG[-1].append(cv2.imread(i,0))
        if timestamp_RFG[-1] == 'X':
            valid_RFG[-1] = False
        runTimestamps_RFG[-1].append(timestamp_RFG.replace('X',''))
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
data_repo_name = "Snowball9"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES'
folder = 'e'
runNames = glob.glob(data_repo_path + os.path.sep +data_folder_name + os.path.sep + folder + os.path.sep + '*')
for i in range(len(runNames)):
    runNames[i] = os.path.basename(runNames[i])
# print(runNames)
runNames = ['control 1'] # the short name of the folder containing images (tif files)
notesContent = []
for runName in runNames:
    Images3 = []
    Images4 = []
    runEventsNoteContent = runName+', Invalid Events: '
    invalidEventsInRun = ''
    detectedFrames = []
    data_folder_name = 'SNOWBALL CROPPED IMAGES'
    data_folder_name += os.path.sep + folder
    data_folder_name += os.path.sep + runName
    data_folder_path = data_repo_path + os.path.sep + data_folder_name # THIS LINE MUST BE CORRECT EVERYTHING ELSE IS NOT ESSENTIAL
    runName, eventPrefixes, eventFrameTimestamps, runEventImages, validRunEvents = getEventsFromRun(data_folder_path) # calls getRunsFromGroup data_folder_path MUST BE A COMPLETE PATH, ALL
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
        eventLabel = str(eventNumber+1)
        if not validRunEvents[eventNumber]:
            runEventsNoteContent+= eventLabel+','
            eventLabel = '('+eventLabel+')'
        thisEventImages = runEventImages[eventNumber]
        thisEventFrameTimestamps = eventFrameTimestamps[eventNumber]
        eventLength = len(thisEventImages)
        if thisEventFrameTimestamps[0][0] == '0':
            thisEventImages.append(thisEventImages.pop(0)) # the 0-th frame is removed and added to the end of the event images
            thisEventFrameTimestamps.append(thisEventFrameTimestamps.pop(0)) # the 0-th frame is removed and added to the end of the event images
        frames = []
        thisEventImages = normalizePixelValues(thisEventImages,30,225) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        Images3.append(np.zeros_like(thisEventImages[0]))
        for frameNumber in range(eventLength):
            frames.append(thisEventImages[frameNumber])
        # thisEventImages = normalizePixelValues(thisEventImages,30,225) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        print(eventLabel)
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
        labels.append(eventLabel)
        theseImages = thisEvent1
        tStamp = []
        for timestamp in thisEventFrameTimestamps:
            tStamp.append(timestamp.replace('.',''))
        # # below is purely comparison basedyy
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
            thisImages.append(theseImages[frameNumber])
            # thisImages.append(cv2.resize(theseImages[frameNumber],(256,96)))
        #     thisImages[frameNumber] = imgNumStamps(int(detectedFrame),7,0,thisImages[frameNumber])
        # thisImages = eventFrameStamp(eventNumber,thisImages,eventPrefixes[eventNumber].replace('.',''),tStamp,True)
        for frameNumber in range(eventLength):
            # thisImages[frameNumber] = imgNumStamps(int(answerKeyLines[eventNumber].split(' ')[0]),20,0,thisImages[frameNumber])
        # leave stamp code output? (seems very useful)
            # if runName == 'control2' or runName == 'control3' or runName == 'old':
            #     Images1.append(cv2.resize(thisImages[frameNumber],(256,96)))
            # else:
            Images.append(thisImages[frameNumber])
            if frameNumber <= detectedFrame + 10 and frameNumber >= detectedFrame:
                Images3[eventNumber]= np.add(Images3[eventNumber],np.divide(thisImages[frameNumber],255))
                # pass
                # Images4.append(thisImages[frameNumber])
                # Images4.append(thisImages[frameNumber])
            # if frameNumber == detectedFrame + 2:
            # Images3[eventNumber]= thisImages[frameNumber]
            if frameNumber < 25:
                Images4.append(frames[frameNumber])
            if frameNumber == eventLength - 1:
                # Images4.append(thisImages[frameNumber])
                pass
        Images3[eventNumber] = np.divide(Images3[eventNumber],np.max([1,np.max(Images3[eventNumber])]))
    makeBarHistGraphsSolo(labels,0.35,codeFrame,runName,False,False,folder+os.path.sep)
    newImage = np.zeros_like(Images3[0])
    newImage1 = np.zeros_like(Images4[0])
    # for i in range(1):
    for i in range(len(Images3)):
        # cv2.imwrite(this_repo_path+os.path.sep+'sandboxFolder'+os.path.sep+str(i+1)+'.jpg',Images3[i])
        newImage = np.add(newImage,np.divide(Images3[i],1))
    # for i in range(len(Images3)):
    #     cv2.imwrite(this_repo_path+os.path.sep+'sandboxFolder'+os.path.sep+str(i+1)+'.jpg',Images3[i])
    #     weighted = np.multiply(Images3[i],[np.rot90([np.arange(len(Images3[i]))]*len(Images3[i][0]),3),[np.arange(len(Images3[i][0]))]*len(Images3[i])])
    #     x,y=np.sum(weighted[0])/np.sum(Images3[i]),np.sum(weighted[1])/np.sum(Images3[i])
    #     print(x,y)
    #     newImage[int(x)][int(y)]+=1
    # for i in range(len(Images3)):
    #     newImage[0][2*i] = i
    #     newImage[1][2*i] = i
    # newImage = np.divide(newImage,len(Images3)/255)
    for i in range(len(Images4)):
        # cv2.imwrite(this_repo_path+os.path.sep+'sandboxFolder'+os.path.sep+str(i+1)+'.jpg',Images4[i])
        # weighted = np.multiply(Images4[i],[np.rot90([np.arange(len(Images4[i]))]*len(Images4[i][0]),3),[np.arange(len(Images4[i][0]))]*len(Images4[i])])
        # x,y=np.sum(weighted[0])/np.sum(Images4[i]),np.sum(weighted[1])/np.sum(Images4[i])
        # print(x,y)
        # newImage1[int(x)][int(y)]+=1
        newImage1 = np.add(newImage1,np.divide(Images4[i],255))
    # print(newImage)
    fig = plt.figure(figsize=(len(Images3[0][0])/8,len(Images3[0])/10))
    plt.clf()
    plt.axis('scaled')
    newImage=concatFrames(newImage,len(Images3)+np.zeros((1,len(newImage[0]))),0)
    y = np.rot90([np.arange(len(newImage))]*(len(newImage[0])),3)
    x = np.array([np.arange(len(newImage[0]))]*(len(newImage)))
    heatmap, xedges, yedges,placeholder = plt.hist2d(x.flatten(), y.flatten(), bins=(len(newImage[0]),len(newImage)),density=False,weights=np.divide(newImage.flatten(),len(Images3)),cmap=plt.cm.nipy_spectral)
    cbar = plt.colorbar()
    cbar.ax.set_xlabel('Density',fontsize=16)
    cbar.ax.tick_params(labelsize=12) 
    plt.xlim(0,len(newImage[0])-1)
    plt.ylim(len(newImage)-2,0)
    plt.xlabel('X (pixels)',fontsize=28)
    plt.ylabel('Y (pixels)',fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    modifyingTitle = 'thisEvent1 10 frames'
    plt.title(folder+' - '+runNames[0]+' - Snowball Nucleation Net Heat Map (N='+str(len(eventPrefixes))+')',fontsize=28)
    fig.tight_layout()
    plt.savefig(this_repo_path+os.path.sep+folder+' - '+runNames[0]+' - heat map ALL - '+modifyingTitle+'.jpg')
    fig = plt.figure(figsize=(len(Images3[0][0])/10*np.ceil(np.sqrt(len(eventPrefixes))),len(Images3[0])/10*np.ceil(np.sqrt(len(eventPrefixes)))))
    plt.clf()
    for eventNumber in range(len(Images3)):
        subPlot = plt.subplot(int(np.ceil(np.sqrt(len(eventPrefixes)))),int(np.ceil(np.sqrt(len(eventPrefixes)))),eventNumber+1)
        subPlot.axis('scaled')
        # subPlot.set_title(str(eventNumber+1),size=50)
        subPlot.text(1,15,str(eventNumber+1),fontsize=100,c='w')
        # plt.subplot(int(np.ceil(np.sqrt(len(eventPrefixes)))),int(np.ceil(np.sqrt(len(eventPrefixes)))),eventNumber+1).set_title('Event '+str(eventNumber))
        # plt.xlabel('X',fontsize=20)
        # plt.ylabel('Y',rotation=0,fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # subPlot.set_xticks(fontsize = 16)
        # subPlot.set_yticks(fontsize = 16)
        newImage=concatFrames(Images3[eventNumber],1+np.zeros((1,len(Images3[eventNumber][0]))),0)
    # newImage=concatFrames(newImage,1+np.zeros((1,len(newImage[0]))),0)
    # newImage=concatFrames(newImage,len(eventPrefixes)+np.zeros((1,len(newImage[0]))),0)
        y = np.rot90([np.arange(len(newImage))]*(len(newImage[0])),3)
        x = np.array([np.arange(len(newImage[0]))]*(len(newImage)))
        heatmap, xedges, yedges,placeholder = plt.hist2d(x.flatten(), y.flatten(), bins=(len(newImage[0]),len(newImage)),density=False,weights=newImage.flatten(),cmap=plt.cm.nipy_spectral)
        # plt.colorbar()
        plt.xlim(0,len(newImage[0])-1)
        plt.ylim(len(newImage)-2,0)
        plt.axis('off')
    plt.suptitle(folder+' - '+runNames[0]+' - Snowball Nucleation Event Heat Maps (N='+str(len(eventPrefixes))+')',fontsize=125,y=.992)
    fig.tight_layout()
    plt.savefig(this_repo_path+os.path.sep+folder+' - '+runNames[0]+' - event heat maps - '+modifyingTitle+'.jpg')
    # plt.show()
    if False:
        newImage1 = np.divide(newImage1,np.max(newImage1)/255)
        # newImage1 = cv2.merge([np.divide(newImage1,1),newImage1,newImage1])#np.zeros_like(newImage1),np.zeros_like(newImage1)])
        # newImage2 = cv2.merge([np.zeros_like(newImage),0*newImage,newImage])
        # newImage = cv2.merge([0*np.mod(np.subtract(255,newImage),255),1/8*np.mod(np.subtract(255,newImage),255),np.mod(0*np.subtract(255,newImage),255)])
        newImage = np.add(np.divide(newImage,1),np.divide(newImage2,1))
        newImage = np.add(np.divide(newImage,5/4),np.divide(newImage1,4))
        newImage = np.divide(newImage,np.max(newImage)/255)
        newImageShape = newImage.shape
        newImage1Shape = newImage1.shape
        SCALE = 4
        newImage = cv2.resize(newImage,(SCALE*newImageShape[1],SCALE*newImageShape[0]))
        newImage1 = cv2.resize(newImage1,(SCALE*newImage1Shape[1],SCALE*newImage1Shape[0]))
        newImage = np.array(newImage).astype(np.uint8)
        newImage1 = np.array(newImage1).astype(np.uint8)
        # while True:
        #     cv2.imshow('test', concatFrames(newImage,newImage1,1))
        #     if cv2.waitKey(0):
        #         cv2.destroyAllWindows()
        #         break
        # while True:
        #     cv2.imshow('test1',newImage)
        #     if cv2.waitKey(0):
        #         cv2.destroyAllWindows()
        #         break
        cv2.imwrite(folder+' - ' + runNames[0]+'0.jpg',newImage)
    # while True:
    #     cv2.imshow('test1',newImage1)
    #     if cv2.waitKey(0):
    #         cv2.destroyAllWindows()
    #         break
    if False:
        txtName = runName
        txtFile = open(this_repo_path+os.path.sep+folder+os.path.sep+txtName+' - Results.txt','w')
        fileContents = "".join(detectedFrames)
        txtFile.write(fileContents)
        txtFile.close()
    notesContent.append(runEventsNoteContent[:-1]+'\n')
# writeAviVideo(videoName ='testVideo',frameRate = 1,allImages = Images3,openVideo = True,color = False)
# writeAviVideo(videoName ='Full Runs - '+folder+' - control0',frameRate = 1,allImages = Images,openVideo = True,color = False)
if False:
    writeAviVideo(videoName = folder+os.path.sep+'Full Runs - '+folder,frameRate = 1,allImages = Images,openVideo = False,color = False)
    txtFile = open(this_repo_path+os.path.sep+folder+os.path.sep+'eventNotes - '+folder+'.txt','w')
    fileContents = "".join(notesContent)
    txtFile.write(fileContents)
    txtFile.close()
    # writeAviVideo(videoName = folder+' - Full Runs 2 of 2',frameRate = 1,allImages = Images1,openVideo = False)
    writeAviVideo(videoName =folder+os.path.sep+'Detection Clips - '+folder,frameRate = 1,allImages = Images2,openVideo = False,color = False)
    nextFilenames = glob.glob(this_repo_path+os.path.sep+folder+os.path.sep+'*.png')
    Images = []
    for filename in nextFilenames:
        print(filename)
        Images.append(cv2.imread(filename.replace(' hist','')))
        Images.append(cv2.imread(filename))
    writeAviVideo(videoName =folder+os.path.sep+ 'Bar Graphs and Histograms Video - '+folder,frameRate = 1/30,allImages= Images,openVideo = False,color = True)