from sklearn import tree
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score
import operator
from matplotlib import pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:10]
y = dataset[:,11]
values = dict([(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)])
for j in range(1,51):
    cvScores = []
    n_components = []
    for n in range(1, 11):
        ica = FastICA(n_components=n)
        icaX = ica.fit_transform(X)
        # X_train, X_test, y_train, y_test = train_test_split(icaX, y, random_state=0, test_size=0.2)
        clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(X_train, y_train)
        scores = cross_val_score(clf, X, y, cv=5)
        cvScores.append(scores.mean())
        n_components.append(n)
    cvScores = np.array(cvScores)
    cvScores = [x * 100 for x in cvScores]
    # print(np.argmax(cvScores))
    values[np.argmax(cvScores)+1] = values[np.argmax(cvScores)+1]+1


# stats = {'a':1000, 'b':3000, 'c': 100}
# print(max(values.iteritems(), key=operator.itemgetter(1))[0])
print(values)
# cvScores = np.array(cvScores)
# cvScores = [x * 100 for x in cvScores]
# print(np.argmax(cvScores))
# print(cvScores)
# print(n_components)
# # print(np.amax(cvScores))
# # print(cvScores[np.argmax(cvScores)])
# # n_components = np.array(n_components)
#
# plt.xlabel("n_components")
# plt.ylabel("cross validation accuracy")
# plt.plot(n_components, cvScores, label='Decision Tree CV accuracy')
# plt.axvline(np.argmax(cvScores)+1,
#             linestyle=':', label='n_components chosen')
# plt.legend(prop=dict(size=12))
# plt.title("Wine Quality ICA")
# plt.show()
#
#








    # print("%0.6f" % (scores.mean()))
    # print(scores.mean())
    # print(clf.score(X_test, y_test))

# confusion_matrix(y_train, clf.predict(X_train))
# print(classification_report(y_train, clf.predict(X_train)))
#
# confusion_matrix(y_test, clf.predict(X_test))
# print(classification_report(y_test, clf.predict(X_test)))

# For each class
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(0,8):
#     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
#                                                         y_score[:, i])
#     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
#     y_score.ravel())
# average_precision["micro"] = average_precision_score(y_test, y_score,
#                                                      average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))
#
