import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


scaler = MinMaxScaler()
output_array=np.ones(10000)
tree_height=[]
isolated_num_array=[]
 
def itree(x,e):
	
	print('e value-'+str(e))
	print('insetted data len to  itree -'+str(len(x)))
	
	
	if len(x)>0:
		ran_nom_1=np.random.randint(low=1, high=len(x), size=1)
		print(int(ran_nom_1))
		Q=x[int(ran_nom_1)]
		print(Q)
		
		ran_nom_2=np.random.randint(low=1, high=len(Q)-1, size=1)
		q=Q[int(ran_nom_2)]
		all_q_values=[]
		
		
		for i in range(len(x)):
			w=x[i]
			k=w.tolist()
			attribute=k[int(ran_nom_2)]
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
		return (xl_np,xr_np,ld,rd,e)
		
		

 
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
		left_,right_,ld,rd,e=itree(d_sample,e)
		e_store(e)
		global tree_height
		if(max(tree_height)<=l):
		  global output_array
		  
		  if(len(left_)>0): 
			 if self.left is None:
				self.left=Node(left_)
				if(len(left_)!=1):
					print('branching for the first left')
					self.left.insert_(left_,e,l)
				if(len(left_)==1):
					print('isolated a left point ')
					isolated_num(left_)
					output_array[ld]=-1
					print(output_array)
					
			    
			 else:
				if(len(left_)!=1):
					print('branching for the second left')
					self.left.insert_(left_,e,l)
				if(len(left_)==1):
					print('isolated a left point ')
					isolated_num(left_)
					output_array[ld]=-1
					print(output_array)
					
		  if(len(right_)>0): 
			 if self.right is None:
				self.right=Node(right_)
				if(len(right_)!=1):
					print('branching for the first right')
					self.right.insert_(right_,e,l)
				if(len(right_)==1):
					print('isolated a right point ')
					isolated_num(right_)
					output_array[rd]=-1
					print(output_array)
					
			    
			 else:
				if(len(right_)!=1):
					print('branching for the second right')
					self.right.insert_(right_,e,l)
				if(len(right_)==1):
					print('isolated a right point ')
					isolated_num(right_)
					output_array[rd]=-1	
					print(output_array)		
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



e=0
l=20

data=normalised_input_data[0:10000]
target_=target[0:10000]

data_withloc=[]
for c in range(len(data)):
		data_withloc.append(np.append(data[c], c))
	
	
tree=BinaryTree(data_withloc)	
tree.root.PrintTree()

samples = 250

sampled_data = list(divide_chunks(data_withloc, samples)) 
print (len(sampled_data)) 

for k in range(len(sampled_data)):
	tree.root.insert_(sampled_data[k],e,l)
	#tree.root.PrintTree()


'''
for k in range(0,10000,1):
	sampled_.append(
	if(k==final_index_no):
		
	
	data_sampled=data_withloc.sample(frac =k) 
	tree.root.insert_(data_sampled,e,l)
	tree.root.PrintTree()

	
'''





#tree=BinaryTree(data_)
#tree.root.PrintTree()


#tree.root.insert_(data_,e,l)
#tree.root.PrintTree()


print(output_array)
print(tree_height)
print(isolated_num_array)


isolated_index=[]
for w in range(len(isolated_num_array)):
	isolated_index.append(int(isolated_num_array[w][0][3]))

print(isolated_index)


for w in range(len(isolated_index)):
	output_array[isolated_index[w]]=-1

print(type(output_array))
#print(target_)



print(type(data))


plt.subplot(2, 1, 1)
plt.scatter(data[:,0], data[:,1],c=output_array, cmap='Paired')
plt.title("IFOREST anormaly detected for the Mulcross data")

plt.subplot(2, 1, 2)
plt.scatter(data[:,0],data[:,1],c=target_, cmap='Paired')
plt.title("Actual target outputs of the Mulcross data")

plt.tight_layout()
plt.show()
