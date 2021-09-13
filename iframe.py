 #!~/.anaconda3/bin/python
import math
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import yaml
import cv2

# WIP
this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + 'control 08 - 8 bit'
data_folder_path = data_repo_path + os.path.sep + data_folder_name
## ------ ##
from PIL import Image
import glob
img_array = []
# for files in glob.glob(data_folder_path + os.path.sep + '*origandcorr.jpg'):
#     try: 
#         os.remove(files)
#     except: pass
run = '3.603154717'
filename = glob.glob(data_folder_path + os.path.sep + run+'*')
for i in filename:
    img = Image.open(i)
    imgarray = np.array(img)
    # imgarray = imgarray.astype(np.float32)
    img_array.append(imgarray)
bkgd = img_array[0]/50
for i in range(49):
    bkgd = bkgd + img_array[i+1]/50
imgs = []
for i in range(len(filename)):
    # print(i)
    imG=img_array[i]
    corr_img=imG-bkgd
    corr_img = np.where(corr_img<0, 0,corr_img)
    img=np.concatenate((imG,corr_img),axis=1)
    iMG=np.array(img).astype(np.uint8)
    height, width = iMG.shape
    size = (width,height)
    imgs.append(iMG)
    
out = cv2.VideoWriter(run+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(imgs)):
    out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
out.release()
from os import startfile
startfile("C:\\Users\\Scott\\Documents\\GitHub\\Snowball2\\"+run+".avi")
# for files in glob.glob(data_folder_path + os.path.sep + '*origandcorr.jpg'):
#     try: 
#         os.remove(files)
#     except: pass


# for i in filename:
    # image_tif = Image.open(filename)
    # imarray = np.array(image_tif)
    # imarrayx=imarray

    # I = Image.fromarray(imarray)
# I.show()
# img_array = []
# for filename in glob.glob(data_folder_path + os.path.sep + '3.603141535*'):
    # img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)


# out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
# from os import startfile
# startfile("C:\\Users\\Scott\\Documents\\GitHub\\Snowball2\\project.mp4")
