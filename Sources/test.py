from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Dropout, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard
from keras.applications import VGG16
import numpy.random as rng
import numpy as np
import os
import sys
import argparse
import dill as pickle
import tensorflow as tf
from sklearn.utils import shuffle
import time

debut = time.time()


def main(args) :   
    print ('##########################\n# MobileNet Neural Network #\n##########################\n')

    #######################
    # Loading of the data #
    #######################

    #Training Data

    print ('Loading of the training data...')

    X_train = pickle.load( open(  args.data_dir + "X_train.p", "rb" ))
    Y_train = pickle.load( open(  args.data_dir + "y_train.p", "rb" ))

    left_data = []
    right_data = []

    n = len(X_train)
    for i in range(n) :
        left_data.append(X_train[i][0])
        right_data.append(X_train[i][1])

    print ('Done.\nThere are {} examples for training.\n'.format(Y_train.shape[0]))

    

    # Validation data

    print ('Loading of the validation data...')

    X_val = pickle.load( open( args.data_dir + "X_val.p", "rb" ))
    Y_val = pickle.load( open(  args.data_dir + "y_val.p", "rb" ))

    validation_left = []
    validation_right = []

    n = len(X_val)
    for i in range(n):
        validation_left.append(X_val[i][0])
        validation_right.append(X_val[i][1])

    print ('Done.\nThere are {} examples for validation.\n'.format(Y_val.shape[0]))

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

    left_input = Input((32,32,1))
    right_input = Input((32,32,1))

    convnet = Conv2D(32, 3, input_shape=(32,32,1))(left_input)
    convnet = MaxPooling2D()(convnet)
    convnet = Conv2D(64, 3, activation='relu')(convnet)
    convnet = MaxPooling2D()(convnet)
    convnet = Conv2D(128,2,activation='relu')(convnet)
    convnet = Flatten()(convnet)
    convnet = Dense(64, activation="relu")(convnet)
    convnet = Dropout(0.5)(convnet)

    encoded_l = convnet
    encoded_r = convnet

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
            epochs=args.epochs,
            verbose=1,
            validation_data=([validation_left,validation_right],Y_val),
            shuffle=args.shuffle,
            callbacks=[tensorboard])

    fin = time.time()
    print("Temps : %f sec" %(fin - debut))

    siamese_net.save(args.logdir + "savedmodel.h5")

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