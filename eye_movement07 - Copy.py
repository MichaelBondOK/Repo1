# 
# https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/
#
# knn for predicting eye state
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
# load the dataset
data = read_csv('c:/anaconda35/mydata/EEG_Eye_State_no_outliers.csv', header=None)


values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
# walk-forward validation
historyX, historyy = [x for x in trainX], [x for x in trainy]
predictions = list()
for i in range(len(testy)):
	# define model
	model = KNeighborsClassifier(n_neighbors=3)
	# fit model on train set
	model.fit(array(historyX), array(historyy))
	# forecast the next time step
	yhat = model.predict([testX[i, :]])[0]
	# store prediction
	predictions.append(yhat)
	# add real observation to history
	historyX.append(testX[i, :])
	historyy.append(testy[i])
# evaluate predictions
score = accuracy_score(testy, predictions)
print(score)



#  We can push this test further and only make the previous 
#  10 observations available to the model when making a prediction.
values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
# walk-forward validation
historyX, historyy = [x for x in trainX], [x for x in trainy]
predictions = list()
for i in range(len(testy)):
	# define model
	model = KNeighborsClassifier(n_neighbors=3)
	# fit model on a small subset of the train set
	tmpX, tmpy = array(historyX)[-10:,:], array(historyy)[-10:]
	model.fit(tmpX, tmpy)
	# forecast the next time step
	yhat = model.predict([testX[i, :]])[0]
	# store prediction
	predictions.append(yhat)
	# add real observation to history
	historyX.append(testX[i, :])
	historyy.append(testy[i])
# evaluate predictions
score = accuracy_score(testy, predictions)
print(score)