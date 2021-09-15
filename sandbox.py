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

this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES' + os.path.sep + 'control 07 - 8 bit'
data_folder_path = data_repo_path + os.path.sep + data_folder_name

# numForms = ['111101101101111','010010010010010','111001111100111','111001111001111','101101111001001','111100111001111','111100111101111','111001001001001','111101111101111','111101111001111']
# grid = np.zeros((7,41),dtype=int)
# for i in range(len(numForms)):
#     for j in range(5):
#         for k in range(3):
#             grid[1+j][1+4*i+k] = numForms[i][3*j+k]
# grid = np.array(grid)
# gridIm = Image.fromarray(np.uint8(255*grid))
# gridIm.show()      
# print(grid)

# grid = np.random.rand(4,2)
# print(grid)
# print(np.min(grid))
# grid-=np.min(grid)
# print(grid)
# grid/=np.max(grid)
# grid*=255
# print(grid)
# run = '3.603138210'
# filename = glob.glob(data_folder_path + os.path.sep +'*.tif')
# print(os.path.split(os.path.split(filename[4])[0])[1])
# groupPath, tifName = os.path.split(filename[4])
# groupName = os.path
# runName, timestamp = tifName.replace('.tif','').split('_')
# # print(runName)
# # print(timestamp)

# def runsFromGroup(groupFolder): # str - name of group, str array - names of runs, str 2d array - timestamps in runs, str 2d array - filenames in runs
#     filename_RFG = glob.glob(groupFolder + os.path.sep + '*.tif')
#     groupName_RFG = os.path.basename(groupFolder)
#     runNames_RFG = []
#     runFilenames_RFG = []
#     runTimestamps_RFG = []
#     currRunName_RFG = ''
#     for i in filename_RFG:
#         tif_RFG = os.path.basename(i)
#         runName_RFG, timestamp_RFG = tif_RFG.replace('.tif','').replace('.','').split('_')
#         if runName_RFG != currRunName_RFG:
#             currRunName_RFG = runName_RFG
#             runNames_RFG.append(runName_RFG)
#             runTimestamps_RFG.append([])
#             runFilenames_RFG.append([])
#         runTimestamps_RFG[-1].append(timestamp_RFG)
#         runFilenames_RFG[-1].append(i)
#     return groupName_RFG, runNames_RFG, runTimestamps_RFG, runFilenames_RFG

# group, runNames, runTimestamps, runFilenames = runsFromGroup(groupPath)
# print(group)
# print(len(runNames))
# for i in runTimestamps:
#     print(len(i))
# print(runFilenames[2][4])

a = [[1,2,3],[4,5,6]]
b = a
print(a[0][0])