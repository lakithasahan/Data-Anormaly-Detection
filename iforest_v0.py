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
scaler = MinMaxScaler()



dfRaw = pd.read_csv('iforest.csv')
x_train = np.array(dfRaw.iloc[:,0:4])
y_train = np.array(dfRaw.iloc[:,4])

target=[]


#print(type(a))
for x in range(len(y_train)):
	if(y_train[x]=="'Normal'"):
		target.append(1)
	else:
		target.append(-1)	


print(target)

scaler.fit(x_train)
normalised_input_data=scaler.transform(x_train)

pca = PCA(n_components=3)
normalised_input_data = pca.fit_transform(normalised_input_data)

scaler.fit(normalised_input_data)
normalised_input_data=scaler.transform(normalised_input_data)


clf = IsolationForest(max_samples=18, random_state=42, contamination=.095)
clf.fit(normalised_input_data)
y_pred = clf.predict(normalised_input_data)





print('precision_score- '+str(precision_score(target,y_pred,average='weighted')))
print('recall_score- '+str(recall_score(target,y_pred,average='weighted')))



plt.subplot(2, 1, 1)
plt.scatter(normalised_input_data[:,0], normalised_input_data[:,1],c=y_pred, cmap='Paired')
plt.title("IFOREST anormaly detected for the Mulcross data")

plt.subplot(2, 1, 2)
plt.scatter(normalised_input_data[:,0], normalised_input_data[:,1],c=target, cmap='Paired')
plt.title("Actual target outputs of the Mulcross data")

plt.tight_layout()
plt.show()









