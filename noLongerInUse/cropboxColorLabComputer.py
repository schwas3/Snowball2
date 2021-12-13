import glob
import os
from typing import Container

import cv2
import keyboard
import numpy as np
from matplotlib.pyplot import figimage, sca, subplots, subplots_adjust

data_folder = 'E:\\Snowball_Data'
orig_folder = data_folder + os.path.sep + 'OriginalsOneForOne'
recep_folder = data_folder + os.path.sep + 'ColorCroppedTiffs'
extraction_folders = os.listdir(orig_folder)
global img,image,x1,x2,y1,y2,scale,index

def useInput(event,x,y,flags,param):
    global x1,x2,y1,y2,rot,img,modifier,scale,index,image
    if event != 0:
        modifier = 0
        if event == cv2.EVENT_LBUTTONDBLCLK:
            event -= cv2.EVENT_LBUTTONDBLCLK
            event += cv2.EVENT_LBUTTONDOWN
            flags -= cv2.EVENT_LBUTTONDBLCLK
            flags += cv2.EVENT_LBUTTONDOWN
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            event -= cv2.EVENT_RBUTTONDBLCLK
            event += cv2.EVENT_RBUTTONDOWN
            flags -= cv2.EVENT_RBUTTONDBLCLK
            flags += cv2.EVENT_RBUTTONDOWN
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY:
                x2 += 1
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                x2 -= 10
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                x2 += 10
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY:
                x2 -= 1
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY:
                x1 -= 1
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY:
                x1 += 10
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY:
                x1 -= 10
            else:
                x1 += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 += 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 -= 10
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 += 10
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY:
                y2 -= 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY:
                y1 -= 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY:
                y1 += 10
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY:
                y1 -= 10
            else:
                y1 += 1
        elif event == cv2.EVENT_MOUSEHWHEEL:
            if flags == 7864320:
                rot -= 0.1
            elif flags == -7864320:
                rot += 0.1
            elif flags == 7864320 + cv2.EVENT_FLAG_CTRLKEY:
                rot -= 1
            elif flags == -7864320 + cv2.EVENT_FLAG_CTRLKEY:
                rot += 1
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags == 7864320:
                scale += .1
            elif flags == -7864320:
                scale -= .1
            elif flags == 7864320 + cv2.EVENT_FLAG_CTRLKEY:
                scale += 1
            elif flags == -7864320 + cv2.EVENT_FLAG_CTRLKEY:
                scale -= 1
            scale = np.max([0.5,scale])
            scale = np.min([10,scale])
        elif event == cv2.EVENT_MBUTTONDOWN:
            if flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_SHIFTKEY:
                index -= 1
            elif flags == cv2.EVENT_FLAG_MBUTTON+ cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY:
                index -= 10
            elif flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_CTRLKEY:
                index += 10
            elif flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY:
                index -= 100
            elif flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_ALTKEY:
                index += 100
            elif flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_CTRLKEY:
                index -= 1000
            elif flags == cv2.EVENT_FLAG_MBUTTON + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_CTRLKEY:
                index += 1000
            else:
                index += 1
            index = np.max([index,0])
            index = np.min([index,len(filenames)-1])
            while True:
                try:image = cv2.imread(filenames[index]);break
                except:index -= 1
        img = np.array(image)
        img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
        img = img[max([0,y1-20]):min([601,y2+20]),max([0,x1-20]):min([801,x2+20]),:]
        # img = rotate_image(image,rot)
        # cv2.imshow('original',cv2.resize(img,(int(scale*(x2-x1+40)),int(scale*(y2-y1+40)))))
        # cv2.imshow('original',img)
        cv2.imshow(sub_path,cv2.resize(img,(int(scale*len(img[0])),int(scale*len(img)))))
        # return np.array([[0,0],[]])
    # return 
theTree = {}
x1,x2,y1,y2 = 300,500,350,550
for run in extraction_folders:
    theTree[run] = {}
    run_path = os.path.join(orig_folder,run)
    recep_run_path = os.path.join(recep_folder,run)
    try:os.mkdir(recep_run_path)
    except:pass
    sub_folders = os.listdir(run_path)
    for sub_folder in sub_folders:
        if sub_folder == '.DS_Store':
            continue
        sub_path = os.path.join(run_path,sub_folder)
        recep_path = os.path.join(recep_folder,run,sub_folder)
        filenames = [sub_path+os.path.sep+filename for filename in os.listdir(sub_path) if filename[0] != 'z']
        # if len(os.listdir(recep_path)) + 2 >= len(filenames):print(sub_folder);continue
        # newFilenames = [os.path.join(recep_path,filename.replace('.bmp','.tiff')) for filename in os.listdir(sub_path) if filename[0] != 'z']
        print(sub_path,len(filenames))
        try:os.mkdir(recep_path)
        except:pass
        # continue
        index = 0
        scale,index,modifier = 4,0,0
        cv2.namedWindow(sub_path)
        cv2.setMouseCallback(sub_path,useInput)
        while modifier < 25:
            try:
                image = cv2.imread(filenames[index])
                img = np.array(image)
                img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
                img = img[max([0,y1-20]):min([601,y2+20]),max([0,x1-20]):min([801,x2+20]),:]
                # cv2.imshow('original',img)
                cv2.imshow(sub_path,cv2.resize(img,(int(scale*len(img[0])),int(scale*len(img)))))
                if cv2.waitKey(0):
                    modifier += 1
            except Exception as excep:
                if excep == KeyboardInterrupt:
                    exit()
                index -= 1
        cv2.destroyAllWindows()
        theTree[run][sub_folder] = [x1*1,x2*1,y1*1,y2*1]
        # print(x1,x2,y1,y2)
        # print(len(filenames),len(filenames)//201)
for run in extraction_folders:
    run_path = os.path.join(orig_folder,run)
    recep_run_path = os.path.join(recep_folder,run)
    sub_folders = os.listdir(run_path)
    for sub_folder in sub_folders:
        if sub_folder == '.DS_Store':continue
        sub_path = os.path.join(run_path,sub_folder)
        recep_path = os.path.join(recep_folder,run,sub_folder)
        filenames = [sub_path+os.path.sep+filename for filename in os.listdir(sub_path) if filename[0] != 'z']
        # if len(os.listdir(recep_path)) + 2 >= len(filenames):print(sub_folder);continue
        print(sub_path,len(filenames))
        newFilenames = [os.path.join(recep_path,filename.replace('.bmp','.tiff')) for filename in os.listdir(sub_path) if filename[0] != 'z']
        x1,x2,y1,y2 = theTree[run][sub_folder]
        print(run,sub_folder,x1,x2,y1,y2)
        for filename,newFilename in zip(filenames,newFilenames):
            # if i % 201 == 0:print(filename,newFilename,i,i//201)
            try:
                # cv2.imread(filename)
                # continue
                cv2.imwrite(newFilename,cv2.imread(filename)[y1:y2,x1:x2,:])
                continue
            except Exception as excep:
                if excep == KeyboardInterrupt:
                    exit()
                cv2.imwrite(newFilename,np.zeros_like(cv2.imread(filenames[index])[y1:y2,x1:x2,:]))
                # continue
                # print(newFilename)
                # cv2.imshow('test',np.zeros((y2-y1,x2-x1)))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
exit()
# global img,image,x1,x2,y1,y2,rot,scale,modifier,index
scale,modifier,index = 1,1,0
image = np.zeros_like(cv2.imread(filenames[index]))
# modifier = 1
# index = 0
# image = cv2.imread(filenames[index])
a = 1
aa = '329 447 416 506 0'
x1,x2,y1,y2,rot = [int(i) for i in aa.split()]
rot /= 10
if a == 1: # image analysis code
    cv2.namedWindow('original')
    image = cv2.imread(filenames[index])
    cv2.setMouseCallback('original',useInput)
    while (1):
        # img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
        img = np.array(image)
        img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
        img = img[max([0,y1-20]):min([601,y2+20]),max([0,x1-20]):min([801,x2+20]),:]
        # img = rotate_image(image,rot)
        # cv2.imshow('original',cv2.resize(img,(int(scale*(x2-x1+40)),int(scale*(y2-y1+40)))))
        cv2.imshow('original',cv2.resize(img,(int(scale*len(img[0])),int(scale*len(img)))))
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()
    print(x1,x2,y1,y2,int(rot*10))
    print(len(filenames))
if a == 2: # make tiffs code
    ii = 0
    for filename in filenames:
        if filename.find('_0') > -1:
            print(str(ii) + ' '+filename)
            ii += 1
        img = image
        # try:
        img = cv2.imread(filename)
        # img = rotate_image(img,rot)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[y1:y2,x1:x2,:]
        # print('pepsi')
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(filename.replace(' Orig',''))
        cv2.imwrite(filename.replace('Snowball7','Snowball8').replace(' Orig',' Tiff').replace('.bmp','.tiff'),img)
        # except:
        #     print('cola',filename)
            # # img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # img = img[y1:y2,x1:x2]
            # cv2.imwrite(filename.split('.bmp')[0]+'X.tiff',img)
