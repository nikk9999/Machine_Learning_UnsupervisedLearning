import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

np.savetxt("PCA_y.csv", y, delimiter=',')

X = PCA(n_components=5).fit_transform(X)
np.savetxt("PCA_reduced.csv",X,delimiter=',')
clusterer = KMeans(n_clusters=8, init='k-means++', random_state=10)

cluster_labels = clusterer.fit_predict(X)
# print(clusterer.cluster_centers_.shape)
# print(cluster_labels.shape)
X_transform = clusterer.fit_transform(X)
np.savetxt("Distances.csv", X_transform, delimiter=',')
# np.savetxt("ClusterLabels.csv", cluster_labels, delimiter=',')

hot_encoding = np.zeros((len(cluster_labels), 8))
for i in range(0,8):
    for j in range(0, len(cluster_labels)):
        print(str(i)+" "+str(j))
        if cluster_labels[j] == i:
            hot_encoding[j][i] = 1
np.savetxt("Cluster_Labels_Hot_Encoding.csv",hot_encoding,delimiter=',')

