from sklearn import tree
import numpy as np
from scipy.stats import kurtosis
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

cvScores = []
n_components = []
kurtosisValues = []

for n in range(1, 11):
    ica = FastICA(n_components=n)
    icaX = ica.fit_transform(X)
    # print(kurtosis(icaX))
    kurtosisValues.append(kurtosis(icaX).mean())
    # X_train, X_test, y_train, y_test = train_test_split(icaX, y, random_state=0, test_size=0.2)
    clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_train, y_train)
    # Y = clf.predict(X_test)
    # print(clf.score(X_test, y_test))
    scores = cross_val_score(clf, X, y, cv=5)
    cvScores.append(scores.mean())
    n_components.append(n)
    # cvScores.append(clf.score(X_test, y_test))
cvScores = np.array(cvScores)
cvScores = [x * 100 for x in cvScores]

n_components = np.array(n_components)
kurtosisValues = np.array(kurtosisValues)

print(np.argmax(cvScores)+1)
print(np.argmax(kurtosisValues)+1)

# fig, ax = plt.subplots()

plt.title("Wine Quality ICA")

fig, ax1 = plt.subplots()

ax1.plot(n_components, kurtosisValues, label='Kurtosis', color='#FFA500')

ax1.set_xlabel('n_components')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Kurtosis', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(n_components, cvScores, 'r', label='Decision Tree CV accuracy')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()

# plt.xlabel("n_components")
# plt.ylabel("Value")
#
# plt.plot(n_components, kurtosisValues, label='Kurtosis',  color='#FFA500')
# plt.axvline(np.argmax(kurtosisValues)+1,
#             linestyle='--', label='kurtosis n_components chosen', color='#FFA500')
# # legend = ax.legend(loc='upper center', shadow=True)
# legend = plt.legend(shadow=True)
# plt.show()
#
# plt.xlabel("n_components")
# plt.ylabel("Value")
# plt.plot(n_components, cvScores, label='Decision Tree CV accuracy')
# plt.axvline(np.argmax(cvScores)+1,
#             linestyle='--', label='Cv n_components chosen')
# plt.legend(prop=dict(size=12))
# plt.title("Wine Quality ICA")
# plt.show()
# ax.show()

#
# print(np.argmax(cvScores))
# print(cvScores)

# ica = FastICA(n_components=3)
# icaX = ica.fit_transform(X)
# print(icaX.shape)
# print(kurtosis(icaX))

# lossA = []
# for n in range(1, 11):
#     pca = FastICA(n_components=n)
#     # pca.fit(X)
#     pca.fit(X)
#     X_train_pca = pca.transform(X)
#     X_projected = pca.inverse_transform(X_train_pca)
#
#     loss = ((X - X_projected) ** 2).mean()
#     print(str(loss))
#     lossA.append(loss)
# plt.plot(np.array(range(1,11)), np.array(lossA), label='reconstruction error')
# plt.show()
# print(n_components)
# # print(np.amax(cvScores))
# # print(cvScores[np.argmax(cvScores)])








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
