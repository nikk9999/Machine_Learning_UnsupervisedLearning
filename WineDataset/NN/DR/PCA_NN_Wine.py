import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:10]
y = dataset[:,11]

X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.4, momentum=0.7,hidden_layer_sizes=(8,2,14), random_state=1, learning_rate='constant')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

