from utils import *
import numpy as np
import random as rn
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Average

def model_1(input: int, output: int):
    input = Input(shape=(input,))
    x = Dense(64, activation = 'relu')(input)
    x = Dense(64, activation = 'relu')(x)
    x =  Dense(32, activation ='sigmoid')(x)
    x = Dropout(0.5)(x)
    output = Dense(output, activation = 'softmax')(x)

    model = Model(inputs=input, outputs=output, name="MODEL1")

    return model

def model_2(input:int, output: int):
    input = Input(shape=(input,))
    x = Dense(32, activation = 'relu')(input)
    x = Dense(16, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(output, activation = 'softmax')(x)

    model = Model(inputs=input, outputs=output, name="MODEL2")

    return model

def model_3(input:int, output: int):
    input = Input(shape=(input,))
    x = Dense(32, activation = 'relu')(input)
    x = Dense(9, activation = 'sigmoid')(x)
    x = Dropout(0.5)(x)
    output = Dense(output, activation = 'softmax')(x)

    model = Model(inputs=input, outputs=output, name="MODEL3")

    return model