import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df=pd.read_csv("/Users/kayttaja/.spyder-py3/fruit_data.csv")

X=np.array(df[["mass","width","height","color_score"]])


y=np.array(pd.get_dummies(df['fruit_name']))

scaler=preprocessing.StandardScaler()
X_scaled=scaler.fit_transform(X)



model = keras.Sequential([
#tf.keras.Input(shape=(2,)

keras.layers.Dense(10, activation=tf.nn.relu, 
                    input_shape=(X_scaled.shape[1],)),
#tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(1,)),
#tf.keras.layers.Dense(20, activation='sigmoid', input_shape=(1,)),

keras.layers.Dense(10, activation=tf.nn.relu),
#tf.keras.layers.Dense(10, activation='sigmoid'),
#tf.keras.layers.Dense(20, activation='sigmoid'),

#tf.keras.layers.Dense(10, activation='relu')
#tf.keras.layers.Dense(20, activation='relu')
keras.layers.Dense(4)    # , activation=tf.nn.softmax)
])

model.compile(loss="categorical_crossentropy",
              optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
              metrics =['categorical_accuracy']) 



model.fit(X_scaled,y, epochs=10 , batch_size=1)
#model.fit(X_scaled,y, epochs=50 , batch_size=1)

#ennuste=model.predict(X_scaled)

ennuste=np.argmax(model.predict(X_scaled), axis=1)

df["Ennuste"]= ennuste







