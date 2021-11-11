import cv2
from matplotlib.pyplot import sca
import numpy as np
import os
import keyboard
import glob

this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball7"
data_repo_path = github_path + os.path.sep + data_repo_name
folder = 'E'
subfolder = 'control 2'
data_folder_path = data_repo_path+os.path.sep+'SNOWBALL CROPPED IMAGES'
data_folder_path += os.path.sep + folder

filenames = glob.glob(data_folder_path+os.path.sep + subfolder+os.path.sep+'*.bmp')
# filenames = glob.glob(data_folder_path+os.path.sep + '*'+os.path.sep+'\u2705.bmp')

# filenames = glob.glob(data_folder_path+os.path.sep+'*'+os.path.sep+'*.bmp')

global img,image,x1,x2,y1,y2,rot,scale,modifier,index
scale,modifier,index = 1,1,0
image = np.zeros_like(cv2.imread(filenames[index]))
# modifier = 1
# index = 0
# image = cv2.imread(filenames[index])

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def useInput(event,x,y,flags,param):
    global x1,x2,y1,y2,rot,img,modifier,scale,index,image
    if event != 0:
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
            image = cv2.imread(filenames[index])
        img = np.array(image)
        img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
        img = img[max([0,y1-20]):min([601,y2+20]),max([0,x1-20]):min([801,x2+20]),:]
        # img = rotate_image(image,rot)
        # cv2.imshow('original',cv2.resize(img,(int(scale*(x2-x1+40)),int(scale*(y2-y1+40)))))
        cv2.imshow('original',cv2.resize(img,(int(scale*len(img[0])),int(scale*len(img)))))
        # return np.array([[0,0],[]])
    # return 
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
if a == 3: #replace code
    for filename in filenames:
        if filename != filename.split(folder)[0]+folder+filename.split(folder)[1].replace(' ',''):
            print(filename)
            print(filename.split(folder)[0]+folder+filename.split(folder)[1].replace(' ',''))
        os.rename(filename,filename.split(folder)[0]+folder+filename.split(folder)[1].replace(' ',''))
if a == 4: # destroy bmp code
    II = 0
    for filename in filenames:
        if filename.find('_0') > -1:
            print(str(II)+ ' '+filename)
            II += 1
        # print(filename)
        os.remove(filename)