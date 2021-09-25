import cv2
from matplotlib.pyplot import sca
import numpy as np
import os
import keyboard
import glob

this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
folder = 'c'
subfolder = 'control1'
data_folder_path = data_repo_path+os.path.sep+'SNOWBALL CROPPED IMAGES'
data_folder_path += os.path.sep + folder

# filenames = glob.glob(data_folder_path+os.path.sep + subfolder+os.path.sep+'*.bmp')

filenames = glob.glob(data_folder_path+os.path.sep+'*'+os.path.sep+'*.bmp')

global img
global image
global x1
global x2
global y1
global y2
global rot
global scale
x1 = 300 #0
x2 = 500 #800
y1 = 400 #0
y2 = 600 #600
rot = 0 #0
scale = 1
global modifier
global index
modifier = 1
index = 0
image = cv2.imread(filenames[index])

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
                x2 -= 50
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                x2 += 50
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY:
                x2 -= 1
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY:
                x1 -= 1
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY:
                x1 += 50
            elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY:
                x1 -= 50
            else:
                x1 += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 += 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 -= 50
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY:
                y2 += 50
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY:
                y2 -= 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY:
                y1 -= 1
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY:
                y1 += 50
            elif flags == cv2.EVENT_RBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY:
                y1 -= 50
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
        img = rotate_image(image,rot)
        cv2.imshow('original',cv2.resize(cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)[y1-20:y2+20,x1-20:x2+20],(int(scale*(x2-x1+40)),int(scale*(y2-y1+40)))))
        # return np.array([[0,0],[]])
    # return 

# cv2.namedWindow('original')
# cv2.setMouseCallback('original',useInput)
# while (1):
#     # img = cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)
#     img = rotate_image(image,rot)
#     cv2.imshow('original',cv2.resize(cv2.rectangle(img,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)[y1-20:y2+20,x1-20:x2+20],(int(scale*(x2-x1+40)),int(scale*(y2-y1+40)))))
#     if cv2.waitKey(0):
#         break
# cv2.destroyAllWindows()
# print(x1,x2,y1,y2,rot,len(filenames))
# x1,x2,y1,y2,rot = 322,436,428,515,0.1

# II = 0
# for filename in filenames:
#     if filename.find('_0') > -1:
#         print(str(II) + ' '+filename)
#         II += 1
#     img = cv2.imread(filename)
#     img = rotate_image(img,rot)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = img[y1:y2,x1:x2]
#     cv2.imwrite(filename.split('.bmp')[0]+'.tiff',img)

II = 0
for filename in filenames:
    if filename.find('_0') > -1:
        print(str(II)+ ' '+filename)
        II += 1
print(filename)
#     os.remove(filename)