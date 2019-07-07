from utils import *
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(42)
rn.seed(554521)
tf.set_random_seed(14452)
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Activation, Input, Average
from keras.models import load_model
import matplotlib as plt

RUN_ENSAMBLE = False

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

def ensamble_model(input: int):
    models = get_models('') 
    models = load_models(models)
    input = Input(shape=(input,))
    eModels =[model(input) for model in models] 
    averageLayer = Average()(eModels)   
    ensModel = Model (name="EnsambleModel", inputs=input, outputs=averageLayer) 

    return ensModel

if __name__ == "__main__":
    trainX, x_test, trainY, y_test = prepare_data()
    output = len(np.unique(y_test))
    model_1 = model_1(trainX.shape[1], output)
    model_1.summary()
    model_1 = fit_model(model_1, trainX,trainY,(x_test, y_test),batch_size=50)
    pred=model_1.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print(pred)
    Y_test2 = np.argmax(y_test, axis=1)
    cm1 =  confusion_matrix(Y_test2, pred)
    np.set_printoptions(precision=2)
    print ("Confusion Matrix1")
    print (cm1)
    
    model_2 = model_2(trainX.shape[1], output)
    # model_2.summary()
    model_2 = fit_model(model_2, trainX,trainY,(x_test, y_test),batch_size=50)
    pred=model_2.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print(pred)
    Y_test2 = np.argmax(y_test, axis=1)
    cm2 =  confusion_matrix(Y_test2, pred)
    np.set_printoptions(precision=2)
    print ("Confusion Matrix2")
    print (cm2)
    model_3 = model_3(trainX.shape[1], output)
    #print(model_3.get_weights())
    # model_3.summary()
    model_3 = fit_model(model_3, trainX,trainY,(x_test, y_test),batch_size=50)
    pred=model_3.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print(pred)
    Y_test2 = np.argmax(y_test, axis=1)
    cm3 =  confusion_matrix(Y_test2, pred)
    np.set_printoptions(precision=2)
    print ("Confusion Matrix3,")
    print (cm3)
   
    
    #model_3.summary()
    print(evalute_model(model_1, x_test, y_test, RUN_ENSAMBLE))
    print(evalute_model(model_2, x_test, y_test, RUN_ENSAMBLE))
    print(evalute_model(model_3, x_test, y_test, RUN_ENSAMBLE))

    save_model(model_1, path='', filename=model_1.name)
    save_model(model_2, path='', filename=model_2.name)
    save_model(model_3, path='', filename=model_3.name)
    
    RUN_ENSAMBLE = True
    if(RUN_ENSAMBLE ==True):
        print("Running ensamble model")
        ensamble = ensamble_model(trainX.shape[1])
        # ensamble.summary()
        ensamble.get_weights()
        print(evalute_model(ensamble, x_test, y_test, RUN_ENSAMBLE))
        pred=ensamble.predict(x_test)
        pred = np.argmax(pred, axis=1)
        Y_test2 = np.argmax(y_test, axis=1)
        cmEnsem =  confusion_matrix(Y_test2, pred)
        np.set_printoptions(precision=2)
        print ("Confusion Matrix ensemble,")
        print (cmEnsem)
        print(pred)