import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

random_state = np.random.RandomState(0)

# Part 1: Quantitative evaluation of various init methods
data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

clf = ExtraTreesClassifier(random_state=4)
clf = clf.fit(X, y)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X_ica = model.transform(X)

plots = []
legends = []
# inertia = []
inertia = []
inertia1 = []
nClusters = []
for n_clusters in range(1, 21):
    km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10).fit(X)
    km1 = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10).fit(X_ica)
    inertia.append(km.inertia_)
    inertia1.append(km1.inertia_)
    nClusters.append(n_clusters)
# legends.append("kmeans++ with %s seed" % ('10'))

inertia = np.array(inertia)
inertia1 = np.array(inertia1)
nClusters = np.array(nClusters)

print(inertia)
# print(nClusters)
print(inertia1)

fig, ax = plt.subplots()

plt.xlabel('n_clusters')
plt.ylabel('sum of square error')

ax.plot(nClusters, inertia1, label='Clustering after FS')
ax.plot(nClusters, inertia, label="Original Clustering")
# ax.set_xticks(nClusters)
# ax.set_yticks(inertia)
# ax.set_yticks(inertia1)
# plt.plot(nClusters, inertia)
legend = ax.legend(shadow=True)
plt.title("Adult FS: K vs Sum of Squared Error")
plt.show()
