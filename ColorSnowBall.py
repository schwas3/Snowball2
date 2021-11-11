 #!~/.anaconda3/bin/python
import math
import numpy as np
import time as t1
import matplotlib.pyplot as plt
import os
import cv2
import glob
from os import read, startfile, write
from os.path import exists
def genPaths(data_folders,run_names):
    os.path.join(
    # -- If the below code does not work, try entering the full path instead --
    this_file_path = os.path.realpath(__file__)
    this_repo_path = os.path.dirname(this_file_path)
    github_path = os.path.dirname(this_repo_path)
    