import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
Xtrain = dataset[:,0:107]
y = dataset[:,108]

pca = PCA()
pca.fit(Xtrain)
lossA = []

for n in range(0, 107):
    pca = PCA(n_components=n)
    pca.fit(Xtrain)

    X_train_pca = pca.transform(Xtrain)
    X_projected = pca.inverse_transform(X_train_pca)

    loss = ((Xtrain - X_projected) ** 2).mean()
    print(str(loss))
    lossA.append(loss)

print(lossA)
plt.title("Adult PCA")
plt.xlabel("n_components")
plt.ylabel("Reconstruction error")
plt.plot(np.array(range(0,107)), np.array(lossA), label='reconstruction error')
plt.show()
