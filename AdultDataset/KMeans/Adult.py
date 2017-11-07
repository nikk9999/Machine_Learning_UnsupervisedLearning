import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

random_state = np.random.RandomState(0)

# Number of run (with randomly generated dataset) for each strategy so as
# to be able to compute an estimate of the standard deviation
n_runs = 5

# k-means models can do several random inits so as to be able to trade
# CPU time for convergence robustness
n_init_range = np.array([1, 5, 10, 15, 20])

# Part 1: Quantitative evaluation of various init methods
data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

plots = []
legends = []
# inertia = []
inertia = []
nClusters = []
for n_clusters in range(1, 21):

    km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10).fit(X)
    inertia.append(km.inertia_)
    nClusters.append(n_clusters)
legends.append("kmeans++ with %s seed" % ('10'))

inertia = np.array(inertia)
nClusters = np.array(nClusters)

print(inertia)
print(nClusters)
print(inertia.shape)
print(nClusters.shape)

# inertia1 = inertia.T
# nClusters1 = nClusters.T
fig, ax = plt.subplots()

plt.xlabel('n_clusters')
plt.ylabel('sum of square error')
plt.legend(plots, legends)

plt.plot(nClusters, inertia, marker='o')
ax.set_xticks(nClusters)
ax.set_yticks(inertia)
plt.title("K vs Sum of Squared Error")
plt.show()
