from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
debut = time.time()

#DATA
X_train = pickle.load( open( "../Data/data_2000/X_train.p", "rb" ))
Y_train = pickle.load( open( "../Data/data_2000/y_train.p", "rb" ))
X_val = pickle.load( open( "../Data/data_2000/X_val.p", "rb" ))
Y_val = pickle.load( open( "../Data/data_2000/y_val.p", "rb" ))

print ('We have {} examples to work with'.format(Y_train.shape[0]-1000))

# We have 2 inputs, 1 for each picture
left_input = Input((32,32,1))
right_input = Input((32,32,1))

# We will use 2 instances of 1 network for this task
convnet = Sequential([
    Conv2D(5,3,activation='relu',input_shape=(32,32,1)),
    MaxPooling2D(),
    Conv2D(5,3,activation='relu'),
    MaxPooling2D(),
    Conv2D(7,2,activation='relu'),
    MaxPooling2D(),
    Conv2D(7,2,activation='relu'),
    Flatten(),
    Dense(18, activation="sigmoid"),
])
# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

left_data = []
right_data = []

n = len(X_train)
for i in range(n) :
    left_data.append(X_train[i][0])
    right_data.append(X_train[i][1])

validation_left = []
validation_right = []
n = len(X_val)
for i in range(n):
    validation_left.append(X_val[i][0])
    validation_right.append(X_val[i][1])

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

siamese_net.summary()
siamese_net.fit([left_data,right_data], Y_train,
          batch_size=16,
          epochs=30,
          verbose=1,
          validation_data=([validation_left,validation_right],Y_val))

fin = time.time()
print("Temps : %f sec" %(fin - debut))
