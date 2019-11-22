import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import  roc_curve,auc
from sklearn.metrics import pairwise_distances,f1_score,precision_score,recall_score
scaler = MinMaxScaler()

tree_height=[]
isolated_num_array=[]
 


def itree(x,e):
	
	print('e value-'+str(e))
	print('insetted data len to  itree -'+str(len(x)))
	
	
	if len(x)>0:
		ran_nom_1=np.random.randint(low=0, high=len(x), size=1)
		print(int(ran_nom_1))
		Q=x[int(ran_nom_1)]
		print(Q)
		
		ran_nom_2=np.random.randint(low=0, high=len(Q)-1, size=1)
		q=Q[int(ran_nom_2)]
		all_q_values=[]
		
		
		for i in range(len(x)):
			w=x[i]
			k=w.tolist()
			attribute=k[int(ran_nom_2)]
			all_q_values.append(attribute)
		
		
		max_random_selected_attribute_value=max(all_q_values)
		min_random_selected_attribute_value=min(all_q_values)
		print('max and min'+str(max_random_selected_attribute_value)+' /'+str(min_random_selected_attribute_value))
		
		
		
		
		p_point_value=np.random.uniform(min_random_selected_attribute_value,max_random_selected_attribute_value)
		print('p value-'+str(p_point_value))
		print('q value -'+str(q))

		left=[]
		right=[]
		for i in range(len(x)):
			w=x[i]
			k=w.tolist()
			q_=k[int(ran_nom_2)]
			if(q_<p_point_value):
				left.append(x[i])
			elif(q_>=p_point_value):
				right.append(x[i])	
		
	
		
		print('left array- '+str(left))
		print('\n')
		print('right array- '+str(right))
		
			
		print(p_point_value)
		print(all_q_values)
		
		
		
		
		if(len(left)<=1):
			print('left last element detected')
			
		if(len(right)<=1):
			print('right last element detected')
			
		
		
		e=e+1
		print('e value-'+str(e))
		return (left,right,e)
		

 
def e_store(e):
	global tree_height
	tree_height.append(e) 
 
def isolated_num(num):
	global isolated_num_array
	isolated_num_array.append(num)
 
class Node(object):
	 
	def __init__(self,d_sample):
		self.d_sample=d_sample
		self.left=None
		self.right=None
	def insert_(self,d_sample,e,l):
		left_,right_,e=itree(d_sample,e)
		e_store(e)
		global tree_height
		if(max(tree_height)<=l):
		
		  
		  if(len(left_)>0): 
			 if self.left is None:
				self.left=Node(left_)
				if(len(left_)!=1):
					print('branching for the first left')
					self.left.insert_(left_,e,l)
				if(len(left_)<=1):
					print('isolated a left point ')
					isolated_num(left_)
					
					
			    
			 else:
				if(len(left_)!=1):
					print('branching for the second left')
					self.left.insert_(left_,e,l)
				if(len(left_)<=1):
					print('isolated a left point ')
					isolated_num(left_)
					
					
		  if(len(right_)>0): 
			 if self.right is None:
				self.right=Node(right_)
				if(len(right_)!=1):
					print('branching for the first right')
					self.right.insert_(right_,e,l)
				if(len(right_)<=1):
					print('isolated a right point ')
					isolated_num(right_)
					
					
					
			    
			 else:
				if(len(right_)!=1):
					print('branching for the second right')
					self.right.insert_(right_,e,l)
				if(len(right_)<=1):
					print('isolated a right point ')
					isolated_num(right_)
						
		else:
			return(0)
			
		 
	def PrintTree(self):
		 if self.left:
		     self.left.PrintTree()
		 print('######'+str(self.d_sample)+'###### \n'),
		 if self.right:
		     self.right.PrintTree()
		 
		 
class BinaryTree(object):
	 def __init__(self,root):
		 self.root=Node(root)		 
		 
		 
		 

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 



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
#print(normalised_input_data)



X_train, X_test, y_train, y_test = train_test_split(normalised_input_data,target, test_size=0.999, random_state=42)



data=[]
for c in range(0,len(X_train),1):
		data.append(np.append(X_train[c], c))
		#print(c)
	
#print('sdsdsdsdsdsd'+str(data))

target_=y_train 						#known target outputs

data=np.asarray(data)
output_array=np.ones(len(data))
print('Total train data length -'+str(len(data)))

samples=300

t=100

e=0

l=math.ceil(math.log(len(data),2))

print('l value generated according to subsample number given -'+str(l))


tree=BinaryTree(data)	


print(data)
sampled_data = list(divide_chunks(data, samples)) 
print ('Total number of subsamples created -'+str(len(sampled_data))) #print(ll)	
#print(ll)

r=0
for k in range(t):
	
	if(k<len(sampled_data)):
		tree.root.insert_(sampled_data[k],e,l)
		r=r+1
	#tree.root.PrintTree()


print(tree_height)
print(isolated_num_array)


isolated_index=[]
for w in range(len(isolated_num_array)):
	isolated_index.append(int(isolated_num_array[w][0][3]))

print(isolated_index)


for w in range(len(isolated_index)):
	output_array[isolated_index[w]]=-1






#print(data)
#for w in range(len(data)):
#	del data[w][0][3]

#print(data)


fpr, tpr, thresholds = roc_curve(target_, output_array)
roc_auc = auc(fpr, tpr)

print('AUC score -'+str(roc_auc))



print('l value -'+str(l)+'subsamples'+str(r))
print('precision_score- '+str(precision_score(target_,output_array,average='weighted')))
print('recall_score- '+str(recall_score(target_,output_array,average='weighted')))




plt.subplot(2, 1, 1)
plt.scatter(data[:,0], data[:,1],c=output_array, cmap='Paired',s=5)
plt.title("IFOREST anormaly detected for the Mulcross data")

plt.subplot(2, 1, 2)
plt.scatter(data[:,0],data[:,1],c=target_, cmap='Paired',s=5)
plt.title("Actual target outputs of the Mulcross data")

plt.tight_layout()
plt.show()
