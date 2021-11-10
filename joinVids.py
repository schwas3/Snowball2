from os import startfile
import cv2
import numpy as np
v1='C:/Users/Scott/Documents/GitHub/Snowball2/Run05/Full Runs - Run05.avi'
v2='C:/Users/Scott/Documents/GitHub/Snowball2/Run05/Full Runs - COLOR - Run05.avi'
cap = cv2.VideoCapture(v1)
i = 0
video = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # thisImg = cv2.imread(frame)
    # cv2.imshow('test',thisImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    video.append(frame)
    i+=1
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(v2)
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    video[i] = np.concatenate((video[i],frame),0)
    i+=1
cap.release()
cv2.destroyAllWindows()
color = cv2.VideoCapture(v2)
height,width,layers = video[0].shape
size = (width,height)
out = cv2.VideoWriter('BWvsColor.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size) # isColor = 0 can be replaced by changing line (this line + 2 (or 3) to out.write(cv2.merge([imgs[i],imgs[i],imgs[i]]))
for i in range(len(video)):
    out.write(video[i])
out.release()
startfile('BWvsColor.avi')