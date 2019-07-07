from utils import *
import numpy as np
import random as rn
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Average
from keras.models import load_model

def create_model(inputs: int, outputs: int, units: [], n_layers: int = 3, 
        dropout: float = 0.5,  activation: str = "relu", output_layer_activation: str = "softmax", name: str = "stacked_model"):

    if (len(units) is not n_layers):
        raise Exception ("The number of layers should match the size of units. E.g units = [25,15, 10] and n_layers = 3")
    
    input = Input(shape=(inputs,)) # Input layer
    hidden_layers = []
    computed_layers = []
    for index, unit in enumerate(units):
        if(index == 0):
            computed_layers.append(Dense(unit, activation = activation)(input))
        else:  
            computed_layers.append(Dense(unit, activation=activation)(computed_layers[index-1]))
       
    x = Dropout(dropout)(computed_layers[-1])
    output = Dense(outputs,activation = output_layer_activation)(x)
    return Model(inputs=input, outputs=output, name=name)
    
if __name__ == "__main__":
    first_model= create_model(15, 3, units=[25, 10], n_layers=2)        
    second_model =create_model(15, 3, units=[45,20, 15, 10], n_layers=4)
    third_model =create_model(15, 3, units=[45,20, 15, 10, 5], n_layers=5)

    first_model.summary()
    second_model.summary()
    third_model.summary()
    first_model.compile()




