import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import random_projection
import numpy as np

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

transformer = random_projection.GaussianRandomProjection(n_components=39)
X_reduced = transformer.fit_transform(X)

np.savetxt("RP.csv", X_reduced, delimiter=',')
np.savetxt("RP_y.csv", y, delimiter=',')

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Adult RP - First three dimensions")
ax.set_xlabel("1st dimension")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd dimension")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd dimension")
ax.w_zaxis.set_ticklabels([])

plt.show()
