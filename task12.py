import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data()

#%%
plt.imshow(x_train[0],cmap='Greys')

#%%
x_train_flat=x_train.reshape(60000,28,28,1)
x_test_flat=x_test.reshape(10000,28,28,1)
x_train_flat=x_train_flat/255
x_test_flat=x_test_flat/255

y_train=np.array(pd.get_dummies(y_train))
y_test=np.array(pd.get_dummies(y_test))

#%%
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(30, kernel_size=5,activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D( pool_size=2, strides=2),
        tf.keras.layers.Conv2D(15, kernel_size=5,activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
        ])

model.compile(loss="categorical_crossentropy",
              optimizers=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
              metrics =['categorical_accuracy']) 

model.fit(x_train_flat,y_train,validation_data=(x_test_flat,y_test) , epochs=10 , batch_size=100)

#%%

ennuste_test=model.predict(x_test_flat)


#plt.imshow(x_test[43],cmap='Greys')
#plt.imshow(x_test[321],cmap='Greys')
#plt.imshow(x_test[231],cmap='Greys')
plt.imshow(x_test[321],cmap='Greys')


#%%
model.fit(x_train_flat,y_train,validation_data=(x_test_flat,y_test) , epochs=10 , batch_size=100)

#%%
model.save('mnistconvmodel.h5')







