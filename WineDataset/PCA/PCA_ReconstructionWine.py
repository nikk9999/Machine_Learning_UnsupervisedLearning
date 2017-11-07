import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
Xtrain = dataset[:,0:10]
y = dataset[:,11]
pca = PCA()
pca.fit(Xtrain)
lossA = []
print(Xtrain.shape)
for n in range(1, 11):
    pca = PCA(n_components=n)
    pca.fit(Xtrain)

    X_train_pca = pca.transform(Xtrain)
    print(X_train_pca.shape)
    X_projected = pca.inverse_transform(X_train_pca)
    print(X_projected.shape)
    loss = ((Xtrain - X_projected) ** 2).mean()
    print(str(loss))
    lossA.append(loss)

# print(lossA)
plt.title("Wine Quality PCA")
plt.xlabel("n_components")
plt.ylabel("Reconstruction error")
plt.plot(np.array(range(1,11)), np.array(lossA), label='reconstruction error')
plt.show()
