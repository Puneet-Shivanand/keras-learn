# Src content from https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("/home/puneet/Documents/keras/keras-learn/data/pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# 1. define the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. fit the network
model.fit(X, Y, nb_epoch=5, batch_size=10)
# 4. evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], accuracy*100))
print("\nLoss:%.2f, Accuracy:%.2f" % (loss, accuracy*100))
# 5. make predictions
probabilities = model.predict(X)
predictions = [numpy.round(x) for x in probabilities]
accuracy = numpy.mean(predictions == Y)
print("\nPrediction Accuracy: %.2f%%" % (accuracy*100))
