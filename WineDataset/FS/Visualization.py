import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:10]
y = dataset[:,11]

clf = ExtraTreesClassifier(random_state=2)
clf = clf.fit(X, y)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X_reduced = model.transform(X)
print(X_reduced.shape[1])

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Wine Gini - First three dimensions")
ax.set_xlabel("1st dimension")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd dimension")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd dimension")
ax.w_zaxis.set_ticklabels([])

plt.show()
