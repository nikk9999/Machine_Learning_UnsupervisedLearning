from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:10]
y = dataset[:,11]

fig, ax = plt.subplots()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
clusterer = KMeans(n_clusters=11, init='k-means++', random_state=10)
clusterer.fit(X)
labels = clusterer.labels_

ax.scatter(X[:, 0], X[:, 1], X[:, 2],
           c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('fixed acidity')
ax.set_ylabel('volatile acidity')
ax.set_zlabel('citric acid')
ax.set_title("Wine Kmeans Clustering")
ax.dist = 12
plt.show()