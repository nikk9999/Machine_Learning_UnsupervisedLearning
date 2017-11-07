import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA
import numpy as np

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = FastICA(n_components=95).fit_transform(X)
np.savetxt("ICA.csv", X_reduced, delimiter=',')
np.savetxt("ICA_y.csv", y, delimiter=',')
ax.scatter(X_reduced[:, 0], X_reduced[:, 10], X_reduced[:, 27], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Adult - First three components")
ax.set_xlabel("1st component")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd component")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd component")
ax.w_zaxis.set_ticklabels([])

plt.show()