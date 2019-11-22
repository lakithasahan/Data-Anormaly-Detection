import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
scaler = MinMaxScaler()
output_array=np.ones(10)

 
def itree(x,e,l):
	
	print('e value-'+str(e))
	print('insetted data len to  itree -'+str(len(x)))
	
	
	
	if len(x)>0:
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
		print('feature min point index -'+str(index_1))
		index_2=all_q_values.index(max((all_q_values)))
		print('feature max point index -'+str(index_2))
		
		if(index_1<index_2):
			split_point=np.random.randint(low=index_1, high=index_2, size=1)
			if(index_1==0):
				split_point=index_1+1
		else:
			split_point=np.random.randint(low=index_2, high=index_1, size=1)
			if(index_2==0):
				split_point=index_2+1
		 
		
		print('split point-' +str(split_point))
		print('\n')
		
		xl=[]
		xr=[]
		
		for i in range(split_point):
				xl.append(x[i])
				print(i)
				
				
		print('\n')
		for j in range(split_point,len(x),1):
				xr.append(x[j])
				print(j)
				
		
		xl_np=np.asarray(xl)
		print('left split array -'+str(xl_np))
		
		xr_np=np.asarray(xr)
		print('right split array -'+str(xr_np))
		#if(len(xl_np)==1
		
		ld=0
		rd=0
		
		if(len(xl_np)<=1):
			print('left last element detected')
			ld=i
		if(len(xr_np)<=1):
			print('right last element detected')
			rd=j	
		
		
		e=e+1
		print('e value-'+str(e))
		return (xl_np,xr_np,ld,rd,)
		
		

 
 
class Node(object):
	 
	 def __init__(self,d_sample):
		 self.d_sample=d_sample
		 self.left=None
		 self.right=None
	 def insert_(self,d_sample,e,l):
		 left_,right_,ld,rd,=itree(d_sample,e,l)
		 global output_array
		 if(len(left_)>0): 
			 if self.left is None:
				self.left=Node(left_)
				if(len(left_)!=1):
					print('branching for the first')
					self.left.insert_(left_,e,l)
				if(len(left_)==1):
					print('isolated a left point ')
					output_array[ld]=-1
					
			    
			 else:
				if(len(left_)!=1):
					print('branching for the second')
					self.left.insert_(left_,e,l)
				if(len(left_)==1):
					print('isolated a left point ')
					output_array[ld]=-1
					
					
		 
			
		 
	 def PrintTree(self):
		 if self.left:
		     self.left.PrintTree()
		 print('######'+str(self.d_sample)+'###### \n'),
		 if self.right:
		     self.right.PrintTree()
		 
		 
class BinaryTree(object):
	 def __init__(self,root):
		 self.root=Node(root)		 
		 
		 
		 






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



tree=BinaryTree(data)
tree.root.PrintTree()

ld=1
rd=0
tree.root.insert_(data2,e,l)




tree.root.PrintTree()


print(output_array)




