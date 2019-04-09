import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam

FILENAME = 'input.csv'
FILENAME1 = 'input2.csv'
RANDOM_SEED = 1000

def save_model(model, path,filename):
    filename = filename + '.h5'
    model.save(path+filename)


def encode_text_index(df, name):
    """
        Label Encoding using sklearn.preporcessing. Transforms labels into integers i.e: [a, b, c] => [1, 2, 3]

        df: pandas.DataFrame
        name: string

    """
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def prepare_data():
    """
        Reads data from file, and splits it into training and testing data

    """
    dataset =  pd.read_csv(FILENAME, sep=',')
    x_data, y_data = to_xy(dataset, '1')
    # import pdb; pdb.set_trace()
    x_data = preprocessing.normalize(x_data)
    trainX, x_test, trainY, y_test =  train_test_split(x_data,y_data,test_size=0.25,random_state=RANDOM_SEED)
    return trainX, x_test, trainY, y_test

def fit_model(model, trainX, trainY,validation_data,batch_size,epochs= 50):
    model.compile(loss='mean_squared_error',  optimizer='Adam', metrics=['acc'])
    scores=model.fit(trainX, trainY, epochs=epochs,batch_size=batch_size,verbose=2, validation_data=validation_data)
    print(scores)
    return model

def evalute_model(model, testX, testY, run_ensamble=False):
    print("EVALUATING MODEL: {0}".format(model.name))
    if(run_ensamble):
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae','acc'])
    score = model.evaluate(testX, testY, verbose = 2)
    return score

def get_models(folder=''):
    if(folder == ''):
        folder = '.'
    else:
        folder = folder + '/'
    
    models =  [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".h5")]

    return models

def load_models(models):
    changed_models = []
    for i in range(len(models)):
        model=load_model(models[i]) 
        changed_models.append(model)
    return changed_models
        
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

#  Transform data to fit the format acceptable by Keras model
def to_xy(df, target):
    result = []
    for x in df.columns:
        # import pdb; pdb.set_trace()
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)