# Voting Ensemble for Classification
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings('ignore')
SEED = 200
dataset =  pd.read_csv('spectral.csv')
#print(dataset.head(30))
x_data=dataset.iloc[:,:-1].values
#print(x_data)
y_data=dataset.iloc[:,-1].values
#print(y_data)
trainX, x_test, trainY, y_test = model_selection.train_test_split(x_data,y_data,test_size=0.25,random_state=SEED)

# create the sub models
estimators = []
model1= MLPClassifier(activation='logistic', alpha=0.05, batch_size=200,
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(100), learning_rate_init=0.01,
              max_iter=500, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=SEED,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
estimators.append(('nn', model1))
model2 = MLPClassifier(activation='relu', alpha=0.05, batch_size=200,
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(80,32),learning_rate_init=0.01,
              max_iter=500, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=SEED,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
estimators.append(('nn1', model2))
model3 = MLPClassifier(activation='tanh', alpha=0.01, batch_size=200,
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(90,45,8),learning_rate_init=0.01,
              max_iter=500, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=SEED,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
estimators.append(('nn2', model3))

for e in estimators:
    print("{0:s} : {1:s}".format( e[0], type(e[1]).__name__))
model_1 = model1.fit(trainX,trainY)
print(model1.score(x_test, y_test))

model_2 = model2.fit(trainX,trainY)
pred2=model_2.predict(x_test)
acc2=accuracy_score(y_test, pred2)
print(acc2)

model_3 = model3.fit(trainX,trainY)
pred3=model_3.predict(x_test)
acc3=accuracy_score(y_test, pred3)
print(acc3)

ensemble = VotingClassifier(estimators)
X=ensemble.fit(trainX, trainY)
predictions = X.predict(x_test)
accuracy1 = accuracy_score(y_test,predictions)
print('Ensemble Accuracy:%.10f' % accuracy1)
print(ensemble.__class__.__name__,accuracy_score(y_test,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

