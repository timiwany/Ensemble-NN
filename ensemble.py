from utils import *
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(42)
rn.seed(554521)
tf.set_random_seed(14452)
from keras.models import Model
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Activation, Input, Average
from keras.models import load_model
from sklearn.metrics import classification_report
import matplotlib as plt

RUN_ENSAMBLE = True
MODEL_SAVE_FOLDER = "models"

def model_1():
    model = Sequential(name="MODEL1")
    model.add(Dense(18, activation = 'sigmoid', input_shape=(28,)))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['mae','acc'])
    return model

def model_2(input:int, output: int):
    input = Input(shape=(input,))
    x = Dense(27, activation = 'relu')(input)
    #x = Dense(7, activation = 'sigmoid')(x)
    x = Dropout(0.1)(x)
    output = Dense(output, activation = 'softmax')(x)
    model = Model(inputs=input, outputs=output, name="MODEL2")

    return model

def model_3(input:int, output: int):
    input = Input(shape=(input,))
    x = Dense(35, activation = 'tanh')(input)
    #x = Dense(10, activation = 'tanh')(x)
    x = Dropout(0.2)(x)
    output = Dense(output, activation = 'softmax')(x)

    model = Model(inputs=input, outputs=output, name="MODEL3")

    return model


def ensamble_model(input: int):
    models = get_models(MODEL_SAVE_FOLDER) 
    models = load_models(models)
    input = Input(shape=(input,))
    eModels =[model(input) for model in models] 
    averageLayer = Average()(eModels)   
    ensModel = Model (name="EnsambleModel", inputs=input, outputs=averageLayer) 

    return ensModel

if __name__ == "__main__":
    trainX,x_test, trainY, y_test = prepare_data('dwt.csv')
    output = len(np.unique(y_test))
    cv_trainX, cv_trainY = get_data_without_encoding('dwt.csv')
    # model_1 = model_1(trainX.shape[1], output)
    train_with_cross_validation(model_1,trainX, trainY,epochs=2)
    import pdb; pdb.set_trace()
    # model_1 = fit_model(model_1, trainX,trainY,epochs=2, k_fold=3)
    
    # model_2 = model_2(trainX.shape[1], output)
    # model_2 = fit_model(model_2, trainX,trainY, epochs=2,k_fold=3)
   
    # model_3 = model_3(trainX.shape[1], output)
    # model_3 = fit_model(model_3, trainX,trainY, epochs=2, k_fold=3)
   
    # print(evalute_model(model_1,trainX, trainY,RUN_ENSAMBLE))
    # print(evalute_model(model_1, x_test, y_test, RUN_ENSAMBLE))
    # T=model_1.predict(x_test)
    # pred = np.argmax(T, axis=1)
    # Y_test = np.argmax(y_test, axis=1)
    # cm =  confusion_matrix(Y_test, pred)
    # np.set_printoptions(precision=2)
    # print ("Confusion Matrix",cm)
    # print(classification_report(Y_test, pred))

    # print(evalute_model(model_2, x_test, y_test, RUN_ENSAMBLE))
    # # print(evalute_model(model_2,trainX, trainY,RUN_ENSAMBLE))
    # T=model_2.predict(x_test)
    # pred = np.argmax(T, axis=1)
   
    # Y_test = np.argmax(y_test, axis=1)
    # cm =  confusion_matrix(Y_test, pred)
    # np.set_printoptions(precision=2)
    # print ("Confusion Matrix",cm)
    # print(classification_report(Y_test, pred))

    # print(evalute_model(model_3, x_test, y_test, RUN_ENSAMBLE))
    # print(evalute_model(model_3,trainX, trainY,RUN_ENSAMBLE))
    # T=model_3.predict(x_test)
    # pred = np.argmax(T, axis=1)
    
    # Y_test = np.argmax(y_test, axis=1)
    # cm =  confusion_matrix(Y_test, pred)
    # np.set_printoptions(precision=2)
    # print ("Confusion Matrix",cm)
    # print(classification_report(Y_test, pred))
    

    save_model(model_1, path=MODEL_SAVE_FOLDER, filename=model_1.name)
    # # save_model(model_2, path=MODEL_SAVE_FOLDER, filename=model_2.name)
    # # save_model(model_3, path=MODEL_SAVE_FOLDER, filename=model_3.name)
    
    # RUN_ENSAMBLE = True
    # if(RUN_ENSAMBLE ==True):
    #     print("Running ensamble model")
    #     ensamble = ensamble_model(trainX.shape[1])
    #     ensamble.summary()
    #     print(evalute_model(ensamble, x_test, y_test, RUN_ENSAMBLE))
    #     pred=ensamble.predict(x_test)
    #     pred = np.argmax(pred, axis=1)
    #     Y_test = np.argmax(y_test, axis=1)
    #     cm =  confusion_matrix(Y_test, pred)
    #     np.set_printoptions(precision=2)
    #     print ("Confusion Matrix")
    #     print (cm)
    #     print(classification_report(Y_test, pred))
        
        