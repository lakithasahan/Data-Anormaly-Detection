import scipy.io as sio
import pandas as pd
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()


from scipy.io import loadmat
x = loadmat('pima.mat')
lon = x['X']
lat = x['y']



#dfRaw = pd.read_csv('iforest.csv')
#x_train = np.array(dfRaw.iloc[:,0:4])
#y_train = np.array(dfRaw.iloc[:,4])

#print(lat)
#print(lon)
#print(y_train)

print(lat[0])

target=lat.flatten()

for  i in range(len(target)):
	if(target[i]==1):
		target[i]=-1
	elif(target[i]==0):
		target[i]=1
	print(target[i])

scaler.fit(lon)
normalised_input_data=scaler.transform(lon)

print(type(normalised_input_data))

clf = IsolationForest(max_samples=100, random_state=42, contamination=.35)
clf.fit(normalised_input_data)
y_pred = clf.predict(normalised_input_data)
accu=0.000
print(list(y_pred).count(-1))
print(len(y_pred))
no_outliers=list(y_pred).count(-1)
l=len(y_pred)
accu=no_outliers/l
print(accu)
print("Accuracy in Detecting Fraud Cases:", accu)


print(y_pred)
print(target)



plt.subplot(2, 1, 1)
plt.scatter(normalised_input_data[:,0], normalised_input_data[:,1],c=y_pred, cmap='Paired')
plt.title("custom DBSCAN predicted cluster outputs")

plt.subplot(2, 1, 2)
plt.scatter(normalised_input_data[:,0], normalised_input_data[:,1],c=target, cmap='Paired')
plt.title("Actual target outputs")

plt.tight_layout()
plt.show()









