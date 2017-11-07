import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]
# clusterer = KMeans(n_clusters=11, init='k-means++', random_state=10)
# cluster_labels = clusterer.fit_predict(X)

gmm = mixture.GaussianMixture(n_components=14,
                                      covariance_type='diag', reg_covar=2E-6)
gmm.fit(X)
cluster_labels = gmm.predict(X)
# print(np.unique(cluster_labels))
colors = ['navy', 'turquoise', 'darkorange', 'r', 'c', 'b', 'g', 'y', 'k', 'm', 'lightsalmon', 'lightpink', 'gold', 'violet' ]
# print(np.unique(cluster_labels))
# print(len(cluster_labels))
for i in range(0,len(cluster_labels)):
    for n, color in enumerate(colors):
        if y[i] == n:
            # print(str(y[i]))
            # print(str(n))
            # print(cluster_labels[i])
            plt.scatter(i, cluster_labels[i], s=8, color=color)
plt.title("Adult EM")
plt.xlabel("Instance Number")
plt.ylabel("Cluster Id")
plt.show()