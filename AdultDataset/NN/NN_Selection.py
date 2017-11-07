import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\AnnNormalized.txt"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:12]
y = dataset[:,13]

X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.5, momentum=0.2,hidden_layer_sizes=(14,3,2), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

data1 = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset1 = np.loadtxt(data1, delimiter=",")
Xdr = dataset1[:,0:107]
ydr = dataset1[:,108]
Xdr_train, Xdr_test, ydr_train, ydr_test = \
                train_test_split(Xdr, ydr, test_size=0.3)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.4, momentum=0.7,hidden_layer_sizes=(8,2,14), random_state=1, learning_rate='constant')
clf.fit(Xdr_train, ydr_train)
print(clf.score(Xdr_test, ydr_test))

