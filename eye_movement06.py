# 
# https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/
#
# knn for predicting eye state
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import flip
# load the dataset
data = read_csv('c:/anaconda35/mydata/EEG_Eye_State_no_outliers.csv', header=None)

values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)
# define model
model = KNeighborsClassifier(n_neighbors=3)
# fit model on train set
model.fit(trainX, trainy)
# forecast test set
yhat = model.predict(testX)
# evaluate predictions
score = accuracy_score(testy, yhat)
print(score)




# Next, we repeat the experiment without shuffling the dataset prior to the split.
values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
# define model
model = KNeighborsClassifier(n_neighbors=3)
# fit model on train set
model.fit(trainX, trainy)
# forecast test set
yhat = model.predict(testX)
# evaluate predictions
score = accuracy_score(testy, yhat)
print(score)





# This is a good start, but not definitive.
# It is possible that the last 10% of the dataset is hard to predict, 
# given the very short open/close intervals we can see on the plot of the outcome variable.
# We can repeat the experiment and use the first 10% of the data in time for
# test and the last 90% for train. 

# load the dataset
data = read_csv('EEG_Eye_State_no_outliers.csv', header=None)
values = data.values
# reverse order of rows
values = flip(values, 0)
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
# define model
model = KNeighborsClassifier(n_neighbors=3)
# fit model on train set
model.fit(trainX, trainy)
# forecast test set
yhat = model.predict(testX)
# evaluate predictions
score = accuracy_score(testy, yhat)
print(score)