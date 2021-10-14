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

other ={
    'five':5
}
newOther = 5
test={'testerer':other,'hold':5,True:0,5:1}
# test['hold'].append(7)
print(test['hold'])
test['new']=8
print(test['new'])
print(test['testerer']['five'])
print(test[test['hold']])