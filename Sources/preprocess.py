import numpy.random as rng
import numpy as np
import os
import sys
import argparse
import dill as pickle


def process_data(X_train, Y_train, X_val, Y_val, preprocess_bool) :
    if preprocess_bool:
        return preprocess_data(X_train, Y_train, X_val, Y_val)
    else :
        return load_data_normally(X_train, Y_train, X_val, Y_val)

def load_data_normally(X_train, Y_train, X_val, Y_val) :
    # Training data
    left_data = []
    right_data = []

    n = len(X_train)

    for i in range(n) :
            left_data.append(X_train[i][0])
            right_data.append(X_train[i][1])

    print ('Done.\nThere are {} examples for training.\n'.format(Y_train.shape[0]))

    # Validation data
    validation_left = []
    validation_right = []

    n = len(X_val)
    for i in range(n):
        validation_left.append(X_val[i][0])
        validation_right.append(X_val[i][1])

    print ('Done.\nThere are {} examples for validation.\n'.format(Y_val.shape[0]))

    return left_data, right_data, validation_left, validation_right


# images normé et d'espérance = 0.5 --> valeurs comprises entre 0 et 1 normalement.
def preprocess_data(X_train, Y_train, X_val, Y_val) :

    # Training data
    left_data = []
    right_data = []

    mean_picture =  [[ [0] for x in range(32)] for y in range(32)]
    std_picture =  [[ [0] for x in range(32)] for y in range(32)]

    n = len(X_train)
    for i in range(n) :
        mean_picture += X_train[i][0]
        mean_picture += X_train[i][1]

    mean_picture = mean_picture / (2*n)

    for i in range(n) :
        std_picture += (X_train[i][0] - mean_picture)**2
        std_picture += (X_train[i][1] - mean_picture)**2

    std_picture = ( std_picture / (2*n))**0.5

    for i in range(n) :
            left_data.append(((X_train[i][0] - mean_picture) / std_picture) + 0.5)
            right_data.append(((X_train[i][1] - mean_picture) / std_picture) + 0.5)

    print ('Done.\nThere are {} examples for training.\n'.format(Y_train.shape[0]))

    # Validation data
    validation_left = []
    validation_right = []

    n = len(X_val)
    for i in range(n):
        validation_left.append(((X_val[i][0] - mean_picture) / std_picture) + 0.5)
        validation_right.append(((X_val[i][1]- mean_picture) / std_picture) + 0.5)

    print ('Done.\nThere are {} examples for validation.\n'.format(Y_val.shape[0]))

    return left_data, right_data, validation_left, validation_right
