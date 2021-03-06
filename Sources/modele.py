from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard
import numpy.random as rng
import numpy as np
import os
import sys
import argparse
import dill as pickle
import tensorflow as tf
from sklearn.utils import shuffle
import time
import preprocess as p

debut = time.time()


def main(args) :
    print ('##########################\n# AlexNet Neural Network #\n##########################\n')

    #######################
    # Loading of the data #
    #######################

    print ('Loading of the training data...')

    X_train = pickle.load( open(  args.data_dir + "X_train.p", "rb" ))
    Y_train = pickle.load( open(  args.data_dir + "y_train.p", "rb" ))

    print ('Loading of the validation data...')

    X_val = pickle.load( open( args.data_dir + "X_val.p", "rb" ))
    Y_val = pickle.load( open(  args.data_dir + "y_val.p", "rb" ))

    #######################
    # Preprocessing Data  #
    #######################

    left_data, right_data, validation_left, validation_right = p.process_data(X_train, Y_train, X_val, Y_val, 0)


    ##################################
    # Creation of the Neural Network #
    ##################################

    #Tensoboard initalization
    date = time.asctime()
    path = args.logdir + "Logs " + date
    reformedPath = path.replace(":", " ")
    try:
        os.mkdir(reformedPath)
    except OSError:
        print ("Creation of the logs directory %s failed" % reformedPath)

    tensorboard = TensorBoard(log_dir=reformedPath, histogram_freq=0,
                              write_graph=True, write_images=True)


    # We have 2 inputs, 1 for each picture, we are currently working with Black and White images in 32x32 pixels.
    size = args.image_size
    dim = args.image_dim
    left_input = Input((size,size,dim))
    right_input = Input((size,size,dim))

    # We will use 2 instances of 1 AlexNet network for this task
    convnet = Sequential([
        Conv2D(5,3,activation='relu',input_shape=(size,size,dim)),
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

    # Getting the L1 Distance between the 2 encodings
    L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network

    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    #Optimiseur : Pour compiler en Keras / https://keras.io/optimizers/ https://towardsdatascience.com/neural-network-optimization-algorithms-1a44c282f61d :
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=2.5e-5) # avant : 0.001, decay=2.5e-4)
    #optimizer = SGD(lr=0.01, momentum=0.0, decay=2.5e-4, nesterov=False)
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    #loss : https://keras.io/losses/
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    ###########################
    # Training and validation #
    ###########################

    siamese_net.summary()
    siamese_net.fit([left_data,right_data], Y_train,
            batch_size=20,
            epochs=100,
            verbose=1,
            validation_data=([validation_left,validation_right],Y_val),
            shuffle= 'true', #args.shuffle,
            callbacks=[tensorboard])

    fin = time.time()
    print("Temps : %f sec" %(fin - debut))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Directory with the training and the validation sets.', default='../Data/data_2000/')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=32)
    parser.add_argument('--image_dim', type=int,
        help='Image dimension (colors) 1 for black and white and 3 for rgb.', default=1)
    parser.add_argument('--epochs', type=int,
        help='Number of epochs.', default=30)
    parser.add_argument('--shuffle',
        help='Shuffle the datasets at each epoch.', action='store_true')
    parser.add_argument('--logdir', type=str,
        help='Directory in which the logs are saved.', default="../logs/")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
