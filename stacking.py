
import pandas as pd
import seaborn as sn
import warnings
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from matplotlib import pyplot
from numpy import dstack
import os 
import shutil 
warnings.filterwarnings('ignore')
SEED = 1000
# define model
def fit_model(trainX, trainY,validation_data,batch_size):
	model = Sequential()
	model.add(Dense(68, input_dim=28, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
	#model.add(Dense(32, activation='sigmoid',kernel_initializer='random_uniform',))
	model.add(Dense(1, activation='sigmoid'))
	Adam=optimizers.Adam(lr=0.01,beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.01, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae','acc'])
	# fit model
	model.fit(trainX, trainY,validation_data=(x_test,y_test),epochs=500, verbose=0)
	#print(model.metrics_names)
	return model
# create directory for models
dataset =  pd.read_csv('dwt.csv')
x_data=dataset.iloc[:,:-1].values
print(x_data)
y_data=dataset.iloc[:,-1].values
print(y_data)
trainX, x_test, trainY, y_test = model_selection.train_test_split(x_data,y_data,test_size=0.30,random_state=SEED)
# create directory for models
dir = 'models'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs('models')

n_members = 3
for i in range(n_members):
	# fit model
	model = fit_model(trainX,trainY,(x_test, y_test),batch_size=100)
	# save model
	filename = 'models/model_' + str(i + 1) + '.h5'
	model.save(filename)
	print('>Saved %s' % filename)
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
# load all models
n_members = 3
smembers = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate the model
for model in members:
	train_acc = model.evaluate(trainX, trainY, verbose=0)
	test_acc = model.evaluate(x_test, y_test, verbose=0)
	print(train_acc, test_acc)
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		y = model.predict(inputX, verbose=2)
		# stack predictions into [rows, members, predictions]
		if stackX is None:
			stackX = y
		else:
			stackX = dstack((stackX, y))
	# fit predictions to [rows, members x predictionss]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

	# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset for the ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit single model
	model = MLPClassifier((40),activation='logistic', solver='adam', alpha=0.5,batch_size=100, max_iter=200,random_state=SEED, tol=1e-10,
     verbose=0,momentum=0.9,nesterovs_momentum=True, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-06)
	model.fit(stackedX, inputy)
	return model
# fit stacked model using the ensemble
model = fit_stacked_model(members, x_test, y_test)
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble+
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	y = model.predict(stackedX)
	return y
# evaluate model on test set
res = stacked_prediction(members, model, x_test)
acc = accuracy_score(y_test, res)
print('Stacked Test Accuracy: %.5f' % acc)
print(confusion_matrix(y_test, res))
