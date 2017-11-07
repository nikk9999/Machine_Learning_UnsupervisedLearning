import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import mixture

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

np.savetxt("PCA_y.csv", y, delimiter=',')
X = PCA(n_components=5).fit_transform(X)
np.savetxt("PCA_reduced.csv",X,delimiter=',')
print(X.shape)
# clusterer = KMeans(n_clusters=14, init='k-means++', random_state=10)

clusterer = mixture.GaussianMixture(n_components=16,
                              covariance_type='full')
clusterer.fit(X)

cluster_labels = clusterer.predict(X)
print(cluster_labels.shape)
print(clusterer.predict_proba(X).shape)

np.savetxt("Cluster_labels.csv", cluster_labels, delimiter=',')
np.savetxt("Cluster_Probabilities.csv", clusterer.predict_proba(X), delimiter=',')

hot_encoding = np.zeros((len(cluster_labels), 16))
for i in range(0, 16):
    for j in range(0, len(cluster_labels)):
        # print(str(i)+" "+str(j))
        if cluster_labels[j] == i:
            hot_encoding[j][i] = 1
np.savetxt("Cluster_Labels_Hot_Encoding.csv",hot_encoding,delimiter=',')



# print(clusterer.weights_.shape)
# print(clusterer.means_.shape)
# print(clusterer.covariances_.shape)
# print(clusterer.precisions_.shape)
# print(clusterer.precisions_cholesky_.shape)
# print(clusterer.score(X).shape)
# print(clusterer.score_samples(X).shape)

#
# # print(clusterer.cluster_centers_.shape)
# # print(cluster_labels.shape)
# X_transform = clusterer.fit_transform(X)
# np.savetxt("Distances.csv", X_transform, delimiter=',')
# # np.savetxt("ClusterLabels.csv", cluster_labels, delimiter=',')
#

