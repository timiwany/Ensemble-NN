import random as rn

import numpy as np
from keras.layers import Activation, Average, Dense, Dropout, Input
from keras.models import Model
from utils import *

RUN_ENSAMBLE = True
def model_1(input: int, output: int):
    model=Sequential()
    input = Input(shape=(input,))
    x = Dense(40, activation = 'relu')(input)
    #x = Dense(9, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(output, activation = 'relu')(x)
    model = Model(inputs=input, outputs=output, name="MODEL1")
    return model

def model_2(input:int, output: int):
    model=Sequential()
    input = Input(shape=(input,))
    x = Dense(15, activation = 'tanh')(input)
    x = Dense(9, activation = 'tanh')(x)
    x = Dropout(0.3)(x)
    output = Dense(output, activation = 'tanh')(x)

    model = Model(inputs=input, outputs=output, name="MODEL2")

    return model

def model_3(input:int, output: int):
    model=Sequential()
    input = Input(shape=(input,))
    x = Dense(50, activation = 'sigmoid')(input)
    x = Dense(32, activation = 'sigmoid')(x)
    x = Dropout(0.2)(x)
    output = Dense(output, activation = 'sigmoid')(x)
    model = Model(inputs=input, outputs=output, name="MODEL3")
    return model