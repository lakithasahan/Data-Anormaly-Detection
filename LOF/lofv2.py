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
import operator
import math
scaler = MinMaxScaler()

np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1


def neighbours(X,k,element_index):
	
	DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'euclidean'))
    index_sort=np.argsort(DistanceMatrix[element_index])
	neighbours=[]
	for j in range(1,k+1,1):
			neighbours.append(index_sort[j])
		
	return(neighbours)	

	
def lof_(X,k,element_index):

	DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'euclidean'))

	element_in_k_distances=[]
	for i in range(len(DistanceMatrix)):
		elements_in_k_range=[]
		distances_=sorted(DistanceMatrix[i])
		for j in range(1,k+1,1):
			
			elements_in_k_range.append(distances_[j])
			
		element_in_k_distances.append(elements_in_k_range[len(elements_in_k_range)-1])

	reach_distance_total=0
	l=0
	
	for w in range(len(element_in_k_distances)):
		
		reach_distance=max(element_in_k_distances[w],DistanceMatrix[element_index][w])	
		
		if(DistanceMatrix[element_index][w]!=0):
			reach_distance_total=reach_distance_total+reach_distance	
			l=l+1

	average_reachdistance=reach_distance_total/l	
	lrd_of_the_element=1/average_reachdistance
	
	return(lrd_of_the_element)	
		


input_data=X[0:220]
lof_calculate_element=60													# Enter the index of the element that need to find lof.
k=20                														#k neighbours



lrd_of_the_element=lof_(input_data,k,lof_calculate_element)					#lrd of the selected element
print('lrd of the selected element -'+str(lrd_of_the_element))

neighbours_indexes=neighbours(input_data,k,lof_calculate_element)			#lrds of the neighbours of the selscted element
print(neighbours_indexes)



lrd_neighbours=[]
for i in range(len(neighbours_indexes)):
	lrd_of_neighbour_element=lof_(input_data,k,neighbours_indexes[i])
	lrd_neighbours.append(lrd_of_neighbour_element)



#print(sum(lrd_neighbours))
#LOF_of_index=lrd_of_the_element/sum(lrd_neighbours)

LOF_of_the_element=np.mean(lrd_neighbours)/lrd_of_the_element

print('LOF of the selected element '+str(lof_calculate_element)+' -'+str(LOF_of_the_element))





