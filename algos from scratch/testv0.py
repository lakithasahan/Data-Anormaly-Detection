import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
scaler = MinMaxScaler()







def itree(x,e,l):
	
	print('Tree height-'+str(e))
	print('x legth -'+str(len(x)))
	
	
	if(e>=l or len(x)<=1):
		return(0,0)
		#return()
	
	else:
		ran_nom_1=np.random.randint(low=1, high=len(x), size=1)
		Q=x[ran_nom_1][0]
		
		ran_nom_2=np.random.randint(low=1, high=len(Q), size=1)
		q=Q[ran_nom_2]
		all_q_values=[]
		
		
		for i in range(len(x)):
			w=x[i]
			k=w.tolist()
			attribute=k[ran_nom_2[0]]
			all_q_values.append(attribute)
		
		
		
		index_1=all_q_values.index(min((all_q_values)))
		index_2=all_q_values.index(max((all_q_values)))
		
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
				
				
		
		xl_np=np.asarray(xl)
		xr_np=np.asarray(xr)
		e=e+1
		
		return (xl_np,xr_np)
		
		












class Node:

    def __init__(self, data,e,l):

        self.left = None
        self.right = None
        self.data = data

    def insert(self, data,e,l):
# Compare the new value with the parent node
		left_,right_=itree(data,e,l)
		self.left=Node(left_,e,l)
		self.right=Node(right_,e,l)
        

# Print the tree
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.data),
        if self.right:
            self.right.PrintTree()




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

data=normalised_input_data[0:10]
data2=normalised_input_data[0:5]
data3=normalised_input_data[5:10]
print(data)
print('###########################')

# Use the insert method to add nodes
root = Node(data,e,l)

d=0
root.insert(data2,e,l)
##d=1
#root.insert(data3,e,l)

#root.insert(data[75:150])


root.PrintTree()
