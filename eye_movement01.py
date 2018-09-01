# 
# https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/
#
#
# visualize dataset
from pandas import read_csv
from matplotlib import pyplot
# load the dataset
data = read_csv('c:/anaconda35/mydata/eye_data.csv', header=None)
# retrieve data as numpy array
values = data.values
# create a subplot for each time series
pyplot.figure()
for i in range(values.shape[1]):
	pyplot.subplot(values.shape[1], 1, i+1)
	pyplot.plot(values[:, i])
pyplot.show()

