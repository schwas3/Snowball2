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
import csv

this_file_path = os.path.realpath(__file__) # gets the path to this file including the file
this_repo_path, this_file_name = os.path.split(this_file_path) # gets the path to the repository containing this file and the file name
github_path, this_repo_name = os.path.split(this_repo_path) # gets the users github folder location and the repo name
## ----- ENTER THE DATA FILE LOCATION INFORMATION HERE ----- ##
data_repo_name = "Snowball3"
data_repo_path = github_path + os.path.sep + data_repo_name
data_folder_name = 'SNOWBALL CROPPED IMAGES'
data_folder_path = data_repo_path + os.path.sep + data_folder_name

filenames = glob.glob(data_folder_path+os.path.sep+'*'+os.path.sep+'known*.txt')
for filename in filenames:
    file = open(filename,'r')
    lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
        if lines[i] != '':
            lines[i] = str(int(lines[i])%201)
        lines[i] += '\n'
    while lines[-1] == '\n':
        lines.pop(-1)
    file = open(filename, 'w')
    newFileContents = "".join(lines)
    file.write(newFileContents)
    file.close()
