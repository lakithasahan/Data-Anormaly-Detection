import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
scaler = MinMaxScaler()



def itree(x,e,l,split_location,random_q_location,data):
	
	print('Tree height-'+str(e))
	print('x legth -'+str(len(x)))
	
	
	if(e>=l or len(x)<=1):
		if(len(x)==1):
			data[random_q_location]=-1
		else:
			data[random_q_location]=1
		print('############')
		print('location--'+str(random_q_location))
		data[random_q_location]=1
	
		return('xlen'+str(len(x)))
		#return()
	
	else:
		ran_nom_1=np.random.randint(low=1, high=len(x), size=1)
		#print(len(x))
		Q=x[ran_nom_1][0]
		#print()
		
		ran_nom_2=np.random.randint(low=1, high=len(Q), size=1)
		#print(ran_nom_2)
		q=Q[ran_nom_2]
		#print('q-'+str(q))
		all_q_values=[]
		
		
		for i in range(len(x)):
			w=x[i]
			k=w.tolist()
			attribute=k[ran_nom_2[0]]
			all_q_values.append(attribute)
		
		
		#print(type(all_q_values))
		#print(max(all_q_values))
		
		index_1=all_q_values.index(min((all_q_values)))
		index_2=all_q_values.index(max((all_q_values)))
		#print(index_1)
		#print(index_2)
		if(index_1<index_2):
			split_point=np.random.randint(low=index_1, high=index_2, size=1)
		else:
			split_point=np.random.randint(low=index_2, high=index_1, size=1)
		#print('p splitpoint index in X-'+str(split_point))
		split_p=x[split_point]
		s_p=split_p[0][ran_nom_2]
		print(split_point)
		print('\n')
		
		xl=[]
		xr=[]
		
		for i in range(split_point-1):
				xl.append(x[i])
				
		
		for i in range(split_point,len(x),1):
				xr.append(x[i])
				
				
		split_location=	split_point
		random_q_location=ran_nom_1
			
		xl_np=np.asarray(xl)
		xr_np=np.asarray(xr)
		e=e+1
		left=itree(xl_np,e,l,split_location,random_q_location,data)
		right=itree(xr_np,e,l,split_location,random_q_location,data)
		#print(e)
		print(lk)
		#print(right)
		



dfRaw = pd.read_csv('iforest.csv')
x_train = np.array(dfRaw.iloc[:,0:4])
y_train = np.array(dfRaw.iloc[:,4])

target=[]


for x in range(len(y_train)):
	if(y_train[x]=="'Normal'"):
		target.append(1)
	else:
		target.append(-1)	


#print(target)

scaler.fit(x_train)
normalised_input_data=scaler.transform(x_train)

pca = PCA(n_components=3)
normalised_input_data = pca.fit_transform(normalised_input_data)

scaler.fit(normalised_input_data)
normalised_input_data=scaler.transform(normalised_input_data)


e=0
l=40

data=np.zeros(len(normalised_input_data))

print(len(normalised_input_data))

a=itree(normalised_input_data,e,l,0,0,data)
#print(a)


