import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing


ennusteaika=12
seqlenght=12

df=pd.read_csv("/Users/kayttaja/.spyder-py3/monthly-car-sales.csv")

df['Month']=pd.to_datetime(df['Month'])
df['Time']=df.index
#%%

df['SalesLag']=df['Sales'].shift(1)
df['SalesDiff']=df.apply(lambda row:
                         row['Sales']-row['SalesLag'], axis=1)

for i in range(1,seqlenght):
    df['SalesDiffLag'+str(i)]=df['SalesDiff'].shift(1)
    
for i in range(1,ennusteaika+1):
    df['SalesDiffFut'+str(i)]=df['SalesDiff'].shift(1)
    
    
df_train=df.iloc[:-2*ennusteaika]
df_train.dropna(inplace=True)
df_test=df.iloc[-2*ennusteaika:]

#%%
input_vars=['SalesDiff']
for i in range(1,seqlenght):
    input_vars.append('SalesDiffLag'+str(i))


output_vars=[]
for i in range(1,ennusteaika+1):
    output_vars.append('SalesDiffFut'+str(i))
    
scaler=preprocessing.StandardScaler()
scalero=preprocessing.StandardScaler()

X=np.array(df_train[input_vars])
X_scaled=scaler.fit_transform(X)

X_scaledLSTM=X_scaled.reshape(X.shape[0],seqlenght,1)
y_scaled=scalero.fit_transform(y)

X_test=np.array(df_test[input_vars])
X_testscaled=scaler.transform(X_test)

X_testscaledLSTM=X_testscaled.reshape(X_test.shape[0],seqlenght,1)



#%%

from sklearn import linear_model

modelLR=linear_model.LinearRegression()
XLR=df_train['Time'].values
XLR=XLR.reshape(-1,1)

yLR=df_train['Sales'].values
yLR=yLR.reshape(-1,1)

modelLR.fit(XLR,yLR)
XLR_test=df_test['Time'].values
XLR_test=XLR_test.reshape(-1,1)

df_test['SalesAvgPred']=modelLR.predict(XLR_test)

#%%
slope=modelLR.coef_


#%%

modelLSTM = tf.keras.Sequential([
      tf.keras.layers.LSTM(24,input_shape=(seqlenght,1),
                          return_sequences=False),
      tf.keras.layers.Dense(ennusteaika)
      ])
      

modelLSTM.compile(loss="mse",
              optimizers=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
              metrics =['mae'])

modelLSTM.fit(X_scaledLSTM,y_scaled, epochs=200 , batch_size=seqlenght )


#%%

ennusteDiff=scalero.inverse_transform(
       modelLSTM.predict(X_testscaledLSTM[ennusteaika-1].reshape(1,12,1)))


ennuste=np.zeros(13)
ennuste[0]=df_test['Sales'][df_test.index[ennusteaika-1]]

for i in range (1,13):
    for j in range(1,13):
        ennuste[j]= ennuste[j-1]+ennusteDiff[0][j-1]+slope()
        
ennuste=np.array(ennuste[1:])























