import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA

data1 = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset1 = np.loadtxt(data1, delimiter=",")
Xdr = dataset1[:,0:107]
ydr = dataset1[:,108]

Xdr_reduced = FastICA(n_components=95).fit_transform(Xdr)

# np.savetxt("ReducedPCA.csv", Xdr_reduced, delimiter=",")
# np.savetxt("ReducedPCA_y.csv", ydr, delimiter=',')

X_train, X_test, y_train, y_test = \
                train_test_split(Xdr_reduced, ydr, test_size=0.3)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.5, momentum=0.2,hidden_layer_sizes=(14,3,2), random_state=1, learning_rate='constant')
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
