from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from numpy import dstack
from models import *
from utils import *

PATH = 'dwt.csv'
SEED = 1000
OUTPUT = 2

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
    stackedX = stacked_dataset(members, inputX)
	# fit single mode
    model = MLPClassifier((40),activation='logistic', solver='adam', alpha=0.5,batch_size=100, max_iter=200,random_state=SEED, tol=1e-10,
     verbose=0,momentum=0.9,nesterovs_momentum=True, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-06)
    
    model.fit(stackedX, inputy)
    return model

def stacked_prediction(members, model, inputX):
	# create dataset using ensemble+
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	y = model.predict(stackedX)
	return y


if __name__ == "__main__":
    trainX, x_test, trainY, y_test = prepare_data(PATH)
    models_to_train = [model_1, model_2, model_3]

    compiled_models = []
    for model in models_to_train:
        model = model(trainX.shape[1], OUTPUT)
        compiled_models.append(fit_model(model,trainX, trainY, (x_test, y_test), batch_size=16, epochs=2))
    for model in compiled_models:
        evalute_model(model, x_test, y_test)
    for model in compiled_models:
        save_model(model,'models', model.name)

    models = get_models('models')
    loaded_models = load_models(models)
    stacked_model = fit_stacked_model(loaded_models, trainX, trainY)
    # evaluate model on test set
    res = stacked_prediction(loaded_models, stacked_model, x_test)
    acc = accuracy_score(y_test, res)
    print('Stacked Test Accuracy: %.5f' % acc)
    # print(confusion_matrix(y_test, res))

