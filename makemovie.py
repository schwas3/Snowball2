 #!~/.anaconda3/bin/python
import math
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
from numpy.ma.core import multiply
import yaml
import cv2
from PIL import Image
import glob
from os import read, startfile, write
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
Images = [] # initializes the array used to store images to make a movie
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
data_repo_name = "Snowball7"
data_repo_path = github_path + os.path.sep + data_repo_name
folder = 'Run05'
subfolder = 'Cs-137 Event'
data_folder_path = data_repo_path+os.path.sep+'SNOWBALL CROPPED IMAGES'
data_folder_path += os.path.sep + folder

filenames = glob.glob(data_folder_path+os.path.sep + subfolder+os.path.sep+'*.bmp')

data_repo_name = "Snowball9"
data_repo_path = github_path + os.path.sep + data_repo_name
folder = 'run05'
subfolder = 'Cs-137'
data_folder_path = data_repo_path+os.path.sep+'SNOWBALL CROPPED IMAGES'
data_folder_path += os.path.sep + folder

filenames2 = glob.glob(data_folder_path+os.path.sep + subfolder+os.path.sep+'3.602044002*.tiff')

# Folder = 'run13'
# data_repo_name = 'Snowball7'
# filenames = glob.glob(github_path + os.path.sep data_repo_name +  + os.path.sep + Folder +os.path.sep+'*.bmp')
# # filenames = glob.glob(this_repo_path + os.path.sep + '*'+os.path.sep+ Folder +os.path.sep+'*.png')
for filename in filenames[1:]+filenames[-1:0]:
    # print(filename)
    # Images.append(cv2.imread(filename.replace(' hist','')))
    Images.append(cv2.imread(filename))
dims = Images[0].shape
print(dims)
Images2=[]
for filename2 in filenames2[1:]+filenames2[-1:0]:
    # print(filename)
    # Images.append(cv2.imread(filename.replace(' hist','')))
    Images2.append(cv2.imread(filename2))
dims2 = Images2[0].shape
print(dims2)
newImages = []
for i in range(len(Images)-1):
    a,b,c,d = Images[i],Images[i+1],Images2[i],Images2[i+1]
    for j in range(10):
        newImages.append(np.concatenate((np.add(np.multiply(a,(10-j)/10),np.multiply(b,j/10)),np.add(np.multiply(c,(10-j)/10),np.multiply(d,j/10))),axis=1))
writeAviVideo(videoName = 'Color vs Gray - Run05 - Cs-137 - 3.602044002',frameRate = 7.5*10,allImages=newImages,openVideo = True,color = True)