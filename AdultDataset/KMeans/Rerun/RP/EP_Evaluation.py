import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import random_projection

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]
transformer = random_projection.GaussianRandomProjection(n_components=25)
X = transformer.fit_transform(X)

silhouette = []
ami = []
homogeneity = []
completeness = []
v_score = []
ari = []
# range_n_clusters = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for n_clusters in range(2,21):

    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette.append(metrics.silhouette_score(X, cluster_labels))
    ari.append(metrics.adjusted_rand_score(y, cluster_labels))
    v_score.append(metrics.v_measure_score(y, cluster_labels))
    ami.append(metrics.adjusted_mutual_info_score(y, cluster_labels))
    homogeneity.append(metrics.homogeneity_score(y, cluster_labels))
    completeness.append(metrics.completeness_score(y, cluster_labels))

range_n_clusters = np.array(range(2,21))
silhouette = np.array(silhouette)
ami = np.array(ami)
homogeneity = np.array(homogeneity)
completeness = np.array(completeness)
v_score = np.array(v_score)
ari = np.array(ari)

fig, ax = plt.subplots()
ax.plot(range_n_clusters, silhouette, 'r', label='Silhouette Coefficient')
ax.plot(range_n_clusters, ari, 'c', label='adjusted_rand_score')
ax.plot(range_n_clusters, ami, 'b', label='adjusted_mutual_info_score')
ax.plot(range_n_clusters, homogeneity, 'g', label='homogeneity')
ax.plot(range_n_clusters, completeness, 'y', label='completeness')
ax.plot(range_n_clusters, v_score, 'm', label='v_measure')
legend = ax.legend(shadow=True)
plt.xlabel('n_clusters')
plt.ylabel('Values')
plt.title('RP KMeans n_clusters vs Metrics')
plt.show()
# plt.savefig(str(n_clusters)+".png")