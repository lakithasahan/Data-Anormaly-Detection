import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import random
from sklearn.metrics import  roc_curve,auc
from sklearn.metrics import pairwise_distances,f1_score,precision_score,recall_score
scaler = MinMaxScaler()


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
		if(len(X)==1):
			isolated_num(X.index)	
		return ExNode(len(X))
    else:
        Q=X.columns
        q=random.choice(Q)
        all_the_q_inX=X[q].unique()
        Max_q=max(all_the_q_inX)
        Min_q=min(all_the_q_inX)
        p=np.random.uniform(Min_q,Max_q)
        X_l=X[X[q]<p]
        X_r=X[X[q]>=p]
        return InNode(iTree(X_l,currHeight+1,hlim),iTree(X_r,currHeight+1,hlim),q,p)




def iForest(X,noOfTrees,sampleSize):
    forest=[]
    hlim=math.ceil(math.log(sampleSize,2))
    for i in range(noOfTrees):
        X_train=X.sample(sampleSize)
        forest.append(iTree(X_train,0,hlim))
  
    return forest


def pathLength(x,Tree,currHeight):
    if isinstance(Tree,ExNode):
        return currHeight
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        return pathLength(x,Tree.left,currHeight+1)
    else:
        return pathLength(x,Tree.right,currHeight+1)



def Average(lst): 
    return (sum(lst)/len(lst)) 
 
def auc_cal(x,y):

	fpr, tpr, thresholds = roc_curve(target_, output_array)
	roc_auc = auc(fpr, tpr)
	return(roc_auc)

dfRaw = pd.read_csv('iforest.csv')

y_train=dfRaw['Target']
x_train=dfRaw.drop('Target',1)


target=[]


for x in range(len(y_train)):
	if(y_train[x]=="'Normal'"):
		target.append(1)
	else:
		target.append(-1)	




scaler.fit(x_train)
normalised_input_data=scaler.transform(x_train)

pca = PCA(n_components=3)
normalised_input_data = pca.fit_transform(normalised_input_data)

scaler.fit(normalised_input_data)
normalised_input_data=scaler.transform(normalised_input_data)




df_input = pd.DataFrame(data=normalised_input_data)

end_length=4096#(normalised_input_data)
start=60000
input_data=df_input[start:start+end_length]
target_=target[start:start+end_length]
output_array=np.ones(end_length)

sampleSize=256								#df.sample(n=1) get a random row from all the data / if n=1000 1000 rows
trees=5
auc=[]
for m in range(0,trees,1):
		
		ifor=iForest(input_data,m,sampleSize) 
		print('Number of isolated numbers -'+str(len(isolated_num_array)))

		for w in range(len(isolated_num_array)):
			output_array[isolated_num_array[w][0]-start]=-1
			
		data_=normalised_input_data[start:start+end_length]
		#fpr, tpr, thresholds = roc_curve(target_, output_array)
		#roc_auc = auc[fpr, tpr]
		#auc.append(roc_auc)
		#roc_auc=auc_cal(target_,output_array)
		#print('AUC score -'+str(roc_auc))
		print('precision_score- '+str(precision_score(target_,output_array,average='weighted')))
		print('f1_score- '+str(f1_score(target_,output_array,average='weighted')))



fpr, tpr, thresholds = roc_curve(target_, output_array)
roc_auc = auc(fpr, tpr)
print(roc_auc)
#auc.append(roc_auc)
'''
t=auc.index(max(auc))

ifor=iForest(input_data,t,sampleSize) 
print('Number of isolated numbers -'+str(len(isolated_num_array)))

for w in range(len(isolated_num_array)):
		
	output_array[isolated_num_array[w][0]-start]=-1


data_=normalised_input_data[0:end_length]


fpr, tpr, thresholds = roc_curve(target_, output_array)
roc_auc = auc(fpr, tpr)
print('Final AUC score -'+str(roc_auc))
print('Final precision_score- '+str(precision_score(target_,output_array,average='weighted')))
print('Final f1_score- '+str(f1_score(target_,output_array,average='weighted')))
'''

normal_p=[]
no_use=[]
anormalies=[]
lengths=[]
all_anormaly_s=[]
for ind in range(0,end_length,1):
	lengths=[]
	for tree in ifor:
		
		lengths.append(pathLength(input_data.iloc[ind],tree,0))
		#print(lengths)
		
	
	#all_length.append(np.mean(lengths))
	#print(len(lengths))
	average_l=np.mean(lengths)
	n=len(input_data)
	cn=2*(np.log(n-1)+0.5772156649)-((2*(n-1))/n)
	#print(cn)
	#print(ll)
	coeficient=-(average_l/cn)
	#print(coeficient)
	anormaly_score=pow(2, coeficient)
	all_anormaly_s.append(anormaly_score)
	#print(anormaly_score)
	
	if(anormaly_score<0.5):
		normal_p.append(ind)
		
	elif(((0.5-0.01)<=anormaly_score )and (anormaly_score<=(0.5+0.01))):
		no_use.appen(ind)
		
	elif anormaly_score>(0.5+0.01):
		anormalies.append(ind)	
	

#print(normal_p)
#print(anormalies)

plt.subplot(2, 1, 1)
plt.scatter(data_[:,0], data_[:,1],c=output_array, cmap='Paired',s=5)
plt.title("IFOREST anormaly detected for the Mulcross data")

plt.subplot(2, 1, 2)
plt.scatter(data_[:,0],data_[:,1],c=target_, cmap='Paired',s=5)
plt.title("Actual target outputs of the Mulcross data")




plt.tight_layout()
plt.show()



