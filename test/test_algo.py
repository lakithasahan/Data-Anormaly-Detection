import pandas as pd
import numpy as np
import math
import random

import random
from matplotlib import pyplot
import os

isolated_num_array=[]
def isolated_num(num):
	global isolated_num_array
	isolated_num_array.append(num)


class ExNode:
    def __init__(self,size):
        self.size=size

class InNode:
    def __init__(self,left,right,splitAtt,splitVal):
        self.left=left
        self.right=right
        self.splitAtt=splitAtt
        self.splitVal=splitVal
        
        
def iTree(X,currHeight,hlim):
    if currHeight>=hlim or len(X)<=1:
		if(len(X)<=1):
			isolated_num(X)
			print('isolted length'+str(len(X)))
			print(X)
			print(ll)
		return ExNode(len(X))
    else:
        Q=X.columns
        print(Q)
        q=random.choice(Q)
        print(q)
        print(X[q])
        print(X[q].unique())
		
        p=random.choice(X[q].unique())
        print(p)
        #print(ll)
        X_l=X[X[q]<p]
        X_r=X[X[q]>=p]
        return InNode(iTree(X_l,currHeight+1,hlim),iTree(X_r,currHeight+1,hlim),q,p)

def iForest(X,noOfTrees,sampleSize):
    forest=[]
    hlim=math.ceil(math.log(sampleSize,2))
    for i in range(noOfTrees):
        X_train=df_data.sample(sampleSize)
        print(X_train)
        forest.append(iTree(X_train,0,hlim))
        print(forest)
        #print(ll)
    return forest

def pathLength(x,Tree,currHeight):
    if isinstance(Tree,ExNode):
        return currHeight
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        return pathLength(x,Tree.left,currHeight+1)
    else:
        return pathLength(x,Tree.right,currHeight+1)

df=pd.read_csv("creditcard.csv")
y_true=df['Class']
df_data=df.drop('Class',1)
df=df[0:100]
sampleSize=10
ifor=iForest(df_data.sample(10),10,sampleSize) ##Forest of 10 trees


print(isolated_num_array)
print(ll)
posLenLst=[]
negLenLst=[]

for sim in range(1000):
    ind=random.choice(df_data[y_true==1].index)
    for tree in ifor:
        posLenLst.append(pathLength(df_data.iloc[ind],tree,0))

    ind=random.choice(df_data[y_true==0].index)
    for tree in ifor:
        negLenLst.append(pathLength(df_data.iloc[ind],tree,0))

bins = np.linspace(0,math.ceil(math.log(sampleSize,2)), math.ceil(math.log(sampleSize,2)))

pyplot.figure(figsize=(12,8))
pyplot.hist(posLenLst, bins, alpha=0.5, label='Anomaly')
pyplot.hist(negLenLst, bins, alpha=0.5, label='Normal')
pyplot.xlabel('Path Length')
pyplot.ylabel('Frequency')
pyplot.legend(loc='upper left')
pyplot.show()
