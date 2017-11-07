from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import mixture

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

np.savetxt("FS_y.csv", y, delimiter=',')

clf = ExtraTreesClassifier(random_state=4)
clf = clf.fit(X, y)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)

np.savetxt("FS_reduced.csv",X,delimiter=',')

clusterer = mixture.GaussianMixture(n_components=10,
                              covariance_type='full')
clusterer.fit(X)

cluster_labels = clusterer.predict(X)

np.savetxt("Cluster_labels.csv", cluster_labels, delimiter=',')
np.savetxt("Cluster_Probabilities.csv", clusterer.predict_proba(X), delimiter=',')

hot_encoding = np.zeros((len(cluster_labels), 10))
for i in range(0, 10):
    for j in range(0, len(cluster_labels)):
        # print(str(i)+" "+str(j))
        if cluster_labels[j] == i:
            hot_encoding[j][i] = 1
np.savetxt("Cluster_Labels_Hot_Encoding.csv",hot_encoding,delimiter=',')

