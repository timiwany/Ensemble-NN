import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

FILENAME = 'dwt.csv'
SEED = 200

def save_model(model, path,filename):
    filename = path+'/'+filename + '.h5'
    model.save(filename)

def encode_text_index(df, name):
    """
        Label Encoding using sklearn.preporcessing. Transforms labels into integers i.e: [a, b, c] => [1, 2, 3]

        df: pandas.DataFrame
        name: string

    """
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def prepare_data(path: str):
   
       # Reads data from file, and splits it into training and testing data

    
    dataset =  pd.read_csv(FILENAME, sep=',', decimal=',')
    print("The last column is {0}".format(dataset.columns[-1]))
    last_column_name = dataset.columns[-1]
    x_data, y_data = to_xy(dataset, last_column_name)
    #pca = PCA(n_components=10)
    #x_data=pca.fit_transform(x_data)
    trainX, x_test, trainY, y_test =  train_test_split(x_data,y_data,test_size=0.25,random_state=47) 
    return trainX, x_test, trainY, y_test

def fit_model(model, trainX, trainY,validation_split,batch_size,epochs, k_fold=True):
    # if (k_fold):
    #     gen = ImageDataGenerator()
    #     # x_train = []
    #     # y_train = []
    #     # folds = generate_fold(trainX, trainY)
    #     # for fold in range(len(folds)):
    #     #     for f in range(fold):
    #     #         x_train.append(trainX[f])
    #     #         y_train.append(trainY[f])
                
    #     generator = gen.flow(trainX, trainY, batch_size=16)
    #     import pdb; pdb.set_trace()
    #     model.fit_generator(generator,epochs=epochs,verbose=1)
    #     return model

    Adam=optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error',  optimizer='Adam', metrics=['acc'])
    scores=model.fit(trainX, trainY, epochs=epochs,batch_size=batch_size,verbose=0, validation_split=validation_split)
    return model

def evalute_model(model, testX, testY, run_ensamble=True):
    print("EVALUATING MODEL: {0}".format(model.name))
    if(run_ensamble):
        Adam=optimizers.Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae','acc'])
    scores = model.evaluate(testX, testY, verbose = 2)
    print(model.metrics_names)
    return scores

def generate_fold(X, Y, k=3):
    folds = list(KFold(n_splits=k, shuffle=True, random_state=1).split(X, Y))
    return folds
    

def get_models(folder=''):
    if(folder == ''):
        folder = '.'
    else:
        folder = folder + '/'
    
    models =  [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".h5")]

    return models

def load_models(models, path="models"):
    changed_models = []
    for i in range(len(models)):
        model=load_model(path+'/'+models[i]) 
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
            result.append(x)s
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