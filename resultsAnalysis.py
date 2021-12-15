import csv
from cv2 import FONT_HERSHEY_COMPLEX, rotate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
resultsDoc = open('resultsDoc2.csv','r')
headings,d = [],[]
x=[]
t=0
fig = plt.figure(figsize=(140/5.63,14))
groupT = ''
g=0
xLabels = []
z=[]
fileQ = ['c*tr*l','ambe','ambe*pb','cs*137','fiesta','UBe','cf*pb']
sourceLabels = ['Control','AmBe Pb','AmBe','Cs-137','UBe','U (\"fiesta\")','Cf Pb','Misc.']
name = [['control','ctrl'], ['AmBe Pb','ambepb','ambe_pb'],['AmBe'], ['Cs137','cs 137','cs-137'], ['UBe'],['fiesta'],['CfPb','Cf Pb'],['filler']]
Colors = ['red','green','blue','yellow','orange','purple','cyan','black']
for i in resultsDoc.readlines()[2::2]:
    i = i.split(',')
    source,group = i[:2]
    if groupT != group:
        g=1-g
        t += 3
        # if group[-2:] == '00':
        #     t=-30
    groupT = group
    # y = [int(j)for j in i[4:]]
    y = [int(j)for j in i[4:]if 0<int(j)<4]
    xLabels.append(group+' - '+source+'('+str(len(y))+')')
    # x = [t]*len(y)
    ymean,yerr = np.mean(y),np.std(y)
    s = 0
    backOut = 0
    for s,n in zip(Colors,name):
        for eachN in n:
            if eachN.lower() in source.lower():
                backOut = 1
                break
        if backOut:break
    print(s)
    plt.scatter(t,ymean,100,s,'s')
    plt.errorbar(t,ymean,yerr,0,elinewidth=5,capsize=5,capthick=5,ecolor=s,linewidth=10,ms=10)
    plt.text(t,ymean+yerr+.1,str(round(ymean,2)),ha='center',fontsize=13)
    plt.text(t,ymean-yerr-.18,str(round(yerr,2)),ha='center',fontsize=13)
    z.append(t)
    t += 1
plt.xticks(z,xLabels,rotation=60,fontsize=12,ha='right')
plt.legend(handles=[mpatches.Patch(color=thisColor,label=aLabel)for thisColor,aLabel in zip(Colors,sourceLabels)],loc=2,fontsize=12)
plt.ylabel('Detected Snowballs',fontsize=20)
plt.xlabel('Folder',fontsize=20)
plt.title('Multiple Scatter Analysis',fontsize=28)
plt.text(t-10,4,'Top Number: Mean\nBottom Number: Std',fontsize=20)
plt.tight_layout()
plt.savefig('resultsAnalysis41.png')