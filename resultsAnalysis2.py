import csv
from cv2 import FONT_HERSHEY_COMPLEX, FONT_HERSHEY_SIMPLEX, rotate
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
data = []
for i in range(len(Colors)):
    data.append([])
print(data)
for i in resultsDoc.readlines()[2::2]:
    i = i.split(',')
    source,group = i[:2]
    s = 0
    backOut = 0
    for s,n in zip(Colors,name):
        for eachN in n:
            if eachN.lower() in source.lower():
                backOut = 1
                break
        if backOut:break
    print(Colors.index(s))
    # data[Colors.index(s)] += [int(j)for j in i[4:]]
    data[Colors.index(s)] += [int(j)for j in i[4:]if 0<int(j)<4]
for t in range(len(Colors)):
    y = data[t]
    ymean,yerr = np.mean(y),np.std(y)
    plt.scatter(t,ymean,100,Colors[t],'s')
    plt.errorbar(t,ymean,yerr,0,elinewidth=5,capsize=5,capthick=5,ecolor=Colors[t],linewidth=10,ms=10)
    plt.text(t,ymean+yerr+.04,str(round(ymean,2)),ha='center',fontsize=13)
    plt.text(t,ymean-yerr-.1,str(round(yerr,2)),ha='center',fontsize=13)
    z.append(t)
plt.xticks(z,sourceLabels,rotation=60,fontsize=20,ha='right')
plt.yticks(fontsize=16)
plt.legend(handles=[mpatches.Patch(color=thisColor,label=aLabel+' (N='+str(len(aData))+')')for thisColor,aLabel,aData in zip(Colors,sourceLabels,data)],loc=2,fontsize=16)
plt.ylabel('Detected Snowballs',fontsize=20)
plt.xlabel('Source',fontsize=20)
plt.title('Multiple Scatter Analysis',fontsize=28)
plt.text(t-1,0,'Top Number: Mean\nBottom Number: Std',fontsize=20)
plt.tight_layout()
plt.savefig('resultsAnalysis42.png')