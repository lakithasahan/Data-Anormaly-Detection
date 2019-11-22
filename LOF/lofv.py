import scipy.io as sio
import pandas as pd
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances,f1_score,precision_score,recall_score
import scipy as scipy
import numpy as np
scaler = MinMaxScaler()





def lof_(X,k,element_index):


	
	DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'euclidean'))
	#print('Distances between each elemets -'+str(DistanceMatrix)+'\n')
	k_distances=[]
	for i in range(len(DistanceMatrix)):
		elements_in_k_range=[]
		distances_=sorted(DistanceMatrix[i])
		#print(distances_)
		for j in range(1,k+1,1):
			elements_in_k_range.append(distances_[j])
		
		
	
		k_distances.append(elements_in_k_range[len(elements_in_k_range)-1])

		
		
	
	reach_distance_total=0
	l=0
	for w in range(len(k_distances)):
		
		reach_distance=max(k_distances[w],DistanceMatrix[element_index][w])	
		
		if(DistanceMatrix[element_index][w]!=0):
			reach_distance_total=reach_distance_total+reach_distance	
			l=l+1

	average_reachdistance=reach_distance_total/l	
	lrd=1/average_reachdistance
	return(lrd)	
		
	


dfRaw = pd.read_csv('iforest.csv')
x_train = np.array(dfRaw.iloc[:,0:4])
y_train = np.array(dfRaw.iloc[:,4])

target=[]


for x in range(len(y_train)):
	if(y_train[x]=="'Normal'"):
		target.append(1)
	else:
		target.append(-1)	


scaler.fit(x_train)
normalised_input_data=scaler.transform(x_train)


input_data=normalised_input_data[0:150]
lof_calculate_element=148  						#0 th element 
k=100                    						#k neighbours



lrd_of_the_element=lof_(input_data,k,lof_calculate_element)
print('lrd of the selected element -'+str(lrd_of_the_element))

other_lrd=[]
for i in range(0,len(input_data),1):
	if(i!=lof_calculate_element):
		lrd_of_the_other_element=lof_(input_data,k,i)
		print('lrd -'+str(lrd_of_the_other_element))
		other_lrd.append(lrd_of_the_other_element)
	

LOF_of_index=lrd_of_the_element/sum(other_lrd)

print('LOF of the selected element -'+str(LOF_of_index))




