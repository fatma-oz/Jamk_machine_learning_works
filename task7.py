import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("/Users/kayttaja/.spyder-py3/fruit_data.csv")

X=np.array(df[["mass","width","height","color_score"]])

koodit={'apple':1,'lemon':0,'mandarin':2,'orange':3}

df['fruit_label']=df['fruit_name'].map(koodit)
y=df['fruit_label']

scaler=preprocessing.StandardScaler()
X_scaled=scaler.fit_transform(X)


model=linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_scaled,y)
ennuste=model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['LRennuste']=ennuste



model=SVC()
model.fit(X_scaled,y)
ennuste=model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['SVMennuste']=ennuste




model=KNeighborsClassifier()
model.fit(X_scaled,y)
ennuste=model.predict(X_scaled)
print(accuracy_score(y, ennuste))
df['KNNennuste']=ennuste

#df1=df[['fruit_label','fruit_name','fruit_subtype',"mass","width","height","color_score"]].head()  
df.head()
