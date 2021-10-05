import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
from sklearn import preprocessing

df=pd.read_csv("/Users/kayttaja/.spyder-py3/Google_Stock_Price.csv")
df["Date"]=pd.to_datetime(df["Date"])
df["Time"]=df.apply(lambda row: len(df)- row.name, axis=1)
df["CloseFuture"]=df["Close"].shift(30)

df_test=df[:185]
df_train=df[185:]

X=np.array(df_train["Time"])
X=X.reshape(-1,1)

scaler=preprocessing.MinMaxScaler()
X_scaled=scaler.fit_transform(X)

y=np.array(df_train["CloseFuture"])



model = tf.keras.Sequential([
#tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(1,)),
##tf.keras.layers.Dense(20, activation='sigmoid', input_shape=(1,)),

#tf.keras.layers.Dense(10, activation='relu'),
tf.keras.layers.Dense(10, activation='sigmoid'),
#tf.keras.layers.Dense(20, activation='sigmoid'),

#tf.keras.layers.Dense(10, activation='relu')
tf.keras.layers.Dense(20, activation='relu'),
tf.keras.layers.Dense(1)
])

model.compile(#optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
              optimizer=tf.compat.v1.train.AdamOptimizer(0.01),
              loss="mse",
              metrics=["mae"])  


model.fit(X_scaled,y, epochs=30 , batch_size=10 )
#model.fit(X_scaled,y, epochs=100 , batch_size=10 )
ennuste_train=model.predict(X_scaled)
df_train["Ennuste"]= ennuste_train

X_test=np.array(df_test["Time"])
X_test=X_test.reshape(-1,1)

X_testscaled=scaler.transform(X_test)
ennuste_test=model.predict(X_testscaled)
df_test["Ennuste"]=ennuste_test

plt.scatter(df["Date"].values,df["Close"], color="black")
plt.plot((df_train["Date"]+pd.DateOffset(days=30)).values,df_train["Ennuste"].values, color="blue")
plt.plot((df_test["Date"]+pd.DateOffset(days=30)).values,df_test["Ennuste"].values, color="red")

plt.show()

df_validation=df_test.dropna()
print("Ennusteen keskivirhe test datassa on %.f"%
      mean_absolute_error(df_validation["CloseFuture"],
                          df_validation["Ennuste"]))



