import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = "C:\\Users\\nikhi\\PycharmProjects\\Assignment3\\AdultDataset\\NN\\EM\\ICA\\ICA_reduced.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:114]
y = dataset[:,115]

X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.1, momentum=0.2,hidden_layer_sizes=(14,3,2), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.2, momentum=0.2,hidden_layer_sizes=16, random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.4, momentum=0.2,hidden_layer_sizes=8, random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.7, momentum=0.2,hidden_layer_sizes=(2,16), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.7, momentum=0.4,hidden_layer_sizes=8, random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.5, momentum=0.6,hidden_layer_sizes=(8,14), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.3, momentum=0.8,hidden_layer_sizes=8, random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.4, momentum=0.7,hidden_layer_sizes=(8,2,14), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.4, momentum=0.7,hidden_layer_sizes=(8,2), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
