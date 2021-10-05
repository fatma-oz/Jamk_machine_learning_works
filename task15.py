import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("/Users/kayttaja/.spyder-py3/Mall_Customers.csv")


X=np.array(df[['Age','Annual Income (k$)', 'Spending Score (1-100)']])

inertia=[]
for i in range(1,14):
    model= KMeans(n_clusters=i)
    model.fit(X)
    inertia.append(model.inertia_)


plt.scatter(np.arange(1,14),inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#%%

model= KMeans(n_clusters=6)
model.fit(X)
labels=model.labels_
df['Label']= labels

#%%


#colors={0:'red',1:'blue',2:'green',3:'magenta'}
colors={0:'red',1:'blue',2:'green',3:'magenta',4:'black',5:'orange'}

fig= plt.figure()
ax=fig.add_subplot(111,projection='3d')

#for i in range(0,4):
for i in range(0,6):
    x=df.loc[df['Label']==i]['Age'].values
    y=df.loc[df['Label']==i]['Annual Income (k$)'].values
    z=df.loc[df['Label']==i]['Spending Score (1-100)'].values
    ax.scatter(x,y,z, marker='o', s=40, color=colors[i],label='Customer class'+str(i+1))
    
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
plt.show()





