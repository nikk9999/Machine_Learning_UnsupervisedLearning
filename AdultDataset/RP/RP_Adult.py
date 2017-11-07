import numpy as np
from sklearn import random_projection
import matplotlib.pyplot as plt

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

n_chosen = []
iterations = [1, 5, 10, 50, 100, 300, 500, 1000]
for j in iterations:
    avg = []
    for n in range(1, 108):
        mse = []
        for i in iterations:
            transformer = random_projection.GaussianRandomProjection(n_components=n)
            transformer = transformer.fit(X)
            comp = transformer.components_ # 30x104
            com_tr = np.transpose(transformer.components_)  # 104x30
            proj = np.dot(X, com_tr)  # 279180x104 * 104x30 = 297180x30
            recon = np.dot(proj, comp)  # 297180x30 * 30x104 = 279180x104
            mse.append(np.mean((X - recon) ** 2))
        avg.append(np.array(mse).mean())
    # print(avg)
    avg = np.array(avg)
    # print(np.argmin(avg)+1)
    n_chosen.append(np.argmin(avg)+1)
print(n_chosen)
n_chosen = np.array(n_chosen)
counts = np.bincount(n_chosen)
print(np.argmax(counts))
np.savetxt("MSE_RP_Adult.csv", avg, delimiter=',')
plt.title("Adult RP")
plt.xlabel("n_components")
plt.ylabel("Mean Reconstruction error")
plt.plot(np.array(range(1, 108)), avg, label='Reconstruction Error')
# plt.axvline(np.argmax(counts), linestyle='--', label='n_components chosen')
legend = plt.legend(shadow=True)
plt.show()