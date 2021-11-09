 #!~/.anaconda3/bin/python
import math
from PIL import Image
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
            images[frameNumber] = imgNumStamps(addLeadingZeros(2,eventNumber+1)+'-'+addLeadingZeros(3,(frameNumber)%len(images)),0,0,images[frameNumber])    
    return images
def normalizePixelValues(eventImages,lowerLimit,upperLimit):
    images = np.array(initializeImages(eventImages))
    images = np.where(images < 0,0,images)
    holder = np.zeros_like(images)
    mid = int(np.mean(images))
    # mid = 240
    # images = cv2.normalize(images,holder,0,255,cv2.NORM_MINMAX)
    # images = np.divide(images,np.mean(images)/255*2)
    if mid > 255/2:
        bottom = np.min([lowerLimit,2*mid-255])
        # bottom = 2*mid-255
        # bottom = 2*mid-255
        images = np.multiply(np.subtract(images,bottom),255/(255-bottom))
        # images = cv2.normalize(images,holder,bottom*255/(bottom-255),255,cv2.NORM_MINMAX)
    else:
        top = np.max([2*mid,upperLimit])
        images = np.multiply(images,255/top)
        # images = cv2.normalize(images,holder,0,255*255/top,cv2.NORM_MINMAX)
    return images
def normalizePixelValues2(eventImages,lowerLimit,upperLimit):
    images = np.array(initializeImages(eventImages))
    holder = np.zeros_like(images)
    mid = int(np.mean(images))
    print(mid)
    # mid = 240
    images = cv2.normalize(images,holder,0,255,cv2.NORM_MINMAX)
    # # images = np.divide(images,np.mean(images)/255*2)
    # if mid > 255/2:
    #     bottom = np.min([lowerLimit,2*mid-255])
    #     print(bottom)
    #     # bottom = 2*mid-255
    #     images = np.multiply(np.subtract(images,bottom),255/(255-bottom))
    #     # images = cv2.normalize(images,holder,bottom*255/(bottom-255),255,cv2.NORM_MINMAX)
    # else:
    #     top = np.max([2*mid,upperLimit])
    #     images = np.multiply(images,255/top)
    #     # images = cv2.normalize(images,holder,0,255*255/top,cv2.NORM_MINMAX)
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
    filename_RFG = [os.path.join(runFolder,i) for i in os.listdir(runFolder)] # make sure is tiff and not .tif possible source of error
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
            runImages_RFG[-1].append(cv2.imread(i,0))
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
data_repo_name = "Snowball9"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES'
folder = 'B'
# runNames = glob.glob(data_repo_path + os.path.sep +data_folder_name + os.path.sep + folder + os.path.sep + '*')
# for i in range(len(runNames)):
#     runNames[i] = os.path.basename(runNames[i])
# print(runNames)
runNames = ['Control1'] # the short name of the folder containing images (tif files)
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
        eventsOfInterest = np.array([7]) # 1-indexing
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
        if not validRunEvents[eventNumber]:
            runEventsNoteContent+= eventLabel+','
            eventLabel = '('+eventLabel+')'
        thisEventImages = runEventImages[eventNumber]
        thisEventFrameTimestamps = eventFrameTimestamps[eventNumber]
        eventLength = len(thisEventImages)
        frames = []
        for frameNumber in range(eventLength):
            frames.append(thisEventImages[frameNumber])
        if thisEventFrameTimestamps[0][0] == '0':
            thisEventImages.append(thisEventImages.pop(0)) # the 0-th frame is removed and added to the end of the event images
            thisEventFrameTimestamps.append(thisEventFrameTimestamps.pop(0)) # the 0-th frame is removed and added to the end of the event images
        thisEventImages = cv2.normalize(np.array(thisEventImages),np.zeros_like(thisEventImages),0,255,cv2.NORM_MINMAX) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        # thisEventImages = cv2.normalize(thisEventImages,50,205) # first number: [0,255/2], second number [255/2,255] 0 and 255 mean no normalization
        Images3.append(np.zeros_like(thisEventImages[0]))
        Images4.append(np.zeros_like(thisEventImages[0]))
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
        backCheck = True
        staticBoy = True
        histLeng = ballParkFrame - 10
        thresh = 100
        blur = 35
        startingAt = 0
        openYorN = True
        theseImages = frames
        # theseImages1 = extractForegroundMask(False,backCheck,staticBoy,thisEventImages,histLeng,thresh,blur,startingAt)
        theseImages1 = extractForegroundMask(False,backCheck,staticBoy,thisEventImages,50,thresh,blur,0)
        theseImages2 = extractForegroundMask(False,False,staticBoy,thisEventImages,50,thresh,blur,0)
        theseImages3 = extractForegroundMask(False,True,True,thisEventImages,ballParkFrame - 10,9,0,ballParkFrame+2) # changed hist from - 10 to - 5 and ballparkframe + 2 from + 5
        # theseImages3 = extractForegroundMask(False,backCheck,staticBoy,thisEventImages,histLeng,9,0,startingAt)
        # modifyingTitle = 'X(false,'+','.join([str(holding).lower() for holding in [backCheck,staticBoy,histLeng,thresh,blur,startingAt]])+')'
        # modifyingTitle = 'X(false,false,true,50,100,15,0)'
        # theseImages = thisEvent1
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
        # thisImages = concatFrames(theseImages,concatFrames(theseImages1,theseImages2,2),2)[:detectedFrame+10]
        # thisImages = concatFrames(concatFrames(concatFrames(concatFrames(theseImages,theseImages2,2),thisEvent2,2),theseImages3,2),thisEvent1,2)[detectedFrame-10:detectedFrame+25]
        thisImages = concatFrames(concatFrames(concatFrames(concatFrames(theseImages,theseImages2,1),thisEvent2,1),theseImages3,1),thisEvent1,1)[detectedFrame-10:detectedFrame+25]
        try:Images = concatFrames(Images,thisImages,0)
        except:Images=thisImages
        # thisImages = concatFrames(theseImages,theseImages1,2)
        # for frameNumber in range(eventLength):
        #     # thisImages.append(cv2.resize(theseImages[frameNumber],(256,96)))
        #     thisImages[frameNumber] = imgNumStamps(int(detectedFrame),7,0,thisImages[frameNumber])
        # thisImages = eventFrameStamp(eventNumber,thisImages,eventPrefixes[eventNumber].replace('.',''),tStamp,True)
        lowVal,highVal = 100,155
        # imagesimages0 = normalizePixelValues(frames,lowVal,highVal) # mine
        # imagesimages1 = normalizePixelValues2(frames,lowVal,highVal) # cv2
        # imagesimages2 = normalizePixelValues(normalizePixelValues2(frames,lowVal,highVal),lowVal,highVal)
        # imagesimages3 = normalizePixelValues(normalizePixelValues2(frames,lowVal,highVal),lowVal,highVal)
        # imagesimages3 = normalizePixelValues2(imagesimages2,lowVal,highVal) # cv2
        # imagesimages3 = normalizePixelValues(normalizePixelValues2(normalizePixelValues(imagesimages2,lowVal,highVal),lowVal,highVal),lowVal,highVal)
        for frameNumber in range(histLeng,startingAt):
            # thisImages[frameNumber] = imgNumStamps(int(answerKeyLines[eventNumber].split(' ')[0]),20,0,thisImages[frameNumber])
        # leave stamp code output? (seems very useful)
            # if runName == 'control2' or runName == 'control3' or runName == 'old':
            #     Images1.append(cv2.resize(thisImages[frameNumber],(256,96)))
            # else:
            # Images.append(concatFrames(concatFrames(theseImages[frameNumber],theseImages1[frameNumber],0),concatFrames(theseImages2[frameNumber],theseImages3[frameNumber],0),0))
            # if frameNumber <= detectedFrame + 200 and frameNumber >= detectedFrame:
            #     Images3[-1]= np.add(Images3[-1],np.divide(theseImages[frameNumber],255))
            #     # Images3[eventNumber]= np.add(Images3[eventNumber],np.divide(thisImages[frameNumber],255))
            # if frameNumber <= detectedFrame + 0 and frameNumber >= detectedFrame:
            #     Images4[-1]= np.add(Images4[-1],np.divide(thisEvent1[frameNumber],255))
                # Images4[eventNumber]= np.add(Images4[eventNumber],np.divide(thisEvent1[frameNumber],255))
                # pass
                # Images4.append(thisImages[frameNumber])
                # Images4.append(thisImages[frameNumber])
            # if frameNumber == detectedFrame + 2:
            # Images3[eventNumber]= thisImages[frameNumber]
            # if frameNumber < 25:
            #     Images4.append(frames[frameNumber])
            # if frameNumber == eventLength - 1:
            #     # Images4.append(thisImages[frameNumber])
            #     pass
        # Images3[-1] = np.divide(Images3[-1],np.max([1,np.max(Images3[-1])]))
        # # Images3[eventNumber] = np.divide(Images3[eventNumber],np.max([1,np.max(Images3[eventNumber])]))
        # Images4[-1] = np.divide(Images4[-1],np.max([1,np.max(Images4[-1])]))
        # Images4[eventNumber] = np.divide(Images4[eventNumber],np.max([1,np.max(Images4[eventNumber])]))
            pass
    continue
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
        # newImage1 = np.add(newImage,np.divide(Images4[i],1))
        # weighted = np.multiply(Images4[i],[np.rot90([np.arange(len(Images4[i]))]*len(Images4[i][0]),3),[np.arange(len(Images4[i][0]))]*len(Images4[i])])
        # x,y=np.sum(weighted[0])/np.sum(Images4[i]),np.sum(weighted[1])/np.sum(Images4[i])
        # print(x,y)
        # newImage1[int(x)][int(y)]+=1
        newImage1 = np.add(newImage1,np.divide(Images4[i],1))
    # print(newImage)
    fig = plt.figure(figsize=(len(Images3[0][0])/8,len(Images3[0])/10))
    plt.clf()
    plt.axis('scaled')
    newImage=concatFrames(newImage,len(Images3)+np.zeros((1,len(newImage[0]))),0)
    y = np.rot90([np.arange(len(newImage))]*(len(newImage[0])),3)
    x = np.array([np.arange(len(newImage[0]))]*(len(newImage)))
    heatmap, xedges, yedges,placeholder = plt.hist2d(x.flatten(), y.flatten(), bins=(len(newImage[0]),len(newImage)),density=False,weights=np.divide(newImage.flatten(),len(Images3)),cmap=plt.cm.nipy_spectral)
    # newImage1 = newImage[:-1]
    # cy = np.sum(np.multiply(y[:-1],newImage1))/np.sum(newImage1)
    # cx = np.sum(np.multiply(x[:-1],newImage1))/np.sum(newImage1)
    # print(cx,cy)
    # plt.hlines(cy,0,len(newImage[0]),colors=['w'])
    # plt.vlines(cx,0,len(newImage),colors=['w'])
    cbar = plt.colorbar()
    cbar.ax.set_xlabel('Density',fontsize=16)
    cbar.ax.tick_params(labelsize=12) 
    plt.xlim(0,len(newImage[0])-1)
    plt.ylim(len(newImage)-2,0)
    plt.xlabel('X (pixels)',fontsize=12)
    plt.ylabel('Y (pixels)',fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(0,len(newImage[0])-1)
    plt.ylim(len(newImage)-2,0)
    # plt.title(folder+' - '+runNames[0]+' - Primary Nucleation Event Location Map N='+str(len(yPos))+'('+str(len(eventPrefixes))+')',fontsize=16)
    fig.tight_layout()
    plt.text(1,len(Images3[0])-5,str(round(np.mean(codeFrame),2))+'+/-'+str(round(np.std(codeFrame),2)),fontsize = 16, c='w')
    plt.title(folder+' - '+runName+' - Snowball Nucleation Net Heat Map (N='+str(len(eventPrefixes))+')',fontsize=16)
    fig.tight_layout()
    plt.savefig(this_repo_path+os.path.sep+folder+' - '+runName+' - heat map ALL - '+modifyingTitle+'.jpg')
    fig = plt.figure(figsize=(len(Images3[0][0])/10*np.ceil(np.sqrt(len(eventPrefixes))),len(Images3[0])/10*np.ceil(np.sqrt(len(eventPrefixes)))))
    plt.clf()
    xPos,yPos,posName=[],[],[]
    for eventNumber in range(len(Images3)):
        subPlot = plt.subplot(int(np.ceil(np.sqrt(numEvents))),int(np.ceil(np.sqrt(numEvents))),eventNumber+1)
        subPlot.axis('scaled')
        # subPlot.set_title(str(eventNumber+1),size=50)
        subPlot.text(1,15,str(eventsOfInterest[eventNumber]+1),fontsize=100,c='w')
        subPlot.text(1,len(Images3[0])-5,detectedFrames[eventNumber].split(',')[0],fontsize=100,c='w')
        # plt.subplot(int(np.ceil(np.sqrt(len(eventPrefixes)))),int(np.ceil(np.sqrt(len(eventPrefixes)))),eventNumber+1).set_title('Event '+str(eventNumber))
        # plt.xlabel('X',fontsize=20)
        # plt.ylabel('Y',rotation=0,fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # subPlot.set_xticks(fontsize = 16)
        # subPlot.set_yticks(fontsize = 16)
        newImage=concatFrames(Images3[eventNumber],1+np.zeros((1,len(Images3[eventNumber][0]))),0)
        newImage1=Images4[eventNumber]
    # newImage=concatFrames(newImage,1+np.zeros((1,len(newImage[0]))),0)
    # newImage=concatFrames(newImage,len(eventPrefixes)+np.zeros((1,len(newImage[0]))),0)
        y = np.rot90([np.arange(len(newImage))]*(len(newImage[0])),3)
        x = np.array([np.arange(len(newImage[0]))]*(len(newImage)))
        if np.sum(newImage1) > 0:
            cy = np.sum(np.multiply(y[:-1],newImage1))/np.sum(newImage1)
            cx = np.sum(np.multiply(x[:-1],newImage1))/np.sum(newImage1)
            yPos.append(cy)
            xPos.append(cx)
            posName.append(str(eventNumber + 1))
            plt.hlines(cy,0,len(newImage[0]),colors=['w'])
            plt.vlines(cx,0,len(newImage),colors=['w'])
        heatmap, xedges, yedges,placeholder = plt.hist2d(x.flatten(), y.flatten(), bins=(len(newImage[0]),len(newImage)),density=False,weights=newImage.flatten(),cmap=plt.cm.nipy_spectral)
        # plt.colorbar()
        # if (eventNumber == 1):
        # print(cx,cy)
        plt.xlim(0,len(newImage[0])-1)
        plt.ylim(len(newImage)-2,0)
        plt.axis('off')
    # xPos = xPos[1:]
    # yPos = yPos[1:]
    # posName = posName[1:]
    plt.suptitle(folder+' - '+runName+' - Snowball Nucleation Event Heat Maps (N='+str(len(eventPrefixes))+')',fontsize=125,y=.992)
    fig.tight_layout()
    plt.savefig(this_repo_path+os.path.sep+folder+' - '+runName+' - event heat maps - '+modifyingTitle+'.jpg')
    # plt.show()
    fig = plt.figure(figsize=(len(Images3[0][0])/8,len(Images3[0])/10))
    plt.clf()
    plt.axis('scaled')
    plt.scatter(xPos,yPos,s=45,marker='.',c='r',label='Primary Nucleation\nEvent Locations')
    # for i in range(len(xPos)):
    #     plt.text(xPos[i],yPos[i],posName[i],fontsize=10,ha='right',va='bottom')
    plt.errorbar([np.mean(xPos)],[np.mean(yPos)],yerr=np.std(yPos),xerr=np.std(xPos),label='X='+str(round(np.mean(xPos),1))+'+/-'+str(round(np.std(xPos),1))+'\nY='+str(round(np.mean(yPos),1))+'+/-'+str(round(np.std(yPos),1)),elinewidth=2,capsize=5,capthick=2,marker='.',ms=15,linewidth=0)
    plt.xlabel('X (pixels)',fontsize=20)
    plt.ylabel('Y (pixels)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,len(newImage[0])-1)
    plt.ylim(len(newImage)-2,0)
    plt.legend(fontsize=16,labelspacing=1)
    plt.title(folder+' - '+runName+' - Primary Nucleation Event Location Map N='+str(len(yPos))+'('+str(len(eventPrefixes))+')',fontsize=24)
    fig.tight_layout()
    plt.savefig(this_repo_path+os.path.sep+folder+' - '+runName+' - event location scatter.jpg')
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
if openYorN:writeAviVideo(videoName = 'TestVideoBacktrackingThisEvent3',frameRate = 7.5,allImages = Images,openVideo = openYorN,color = False)
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