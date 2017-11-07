import numpy as np
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn import random_projection

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

# X_pca = PCA(n_components=3).fit_transform(X)
transformer = random_projection.GaussianRandomProjection(n_components=25)
X_pca = transformer.fit_transform(X)

lowest_bic = np.infty
lowest_bic1 = np.infty
bic = []
bic1 = []
n_components_range = range(1, 11)
cv_types = ['spherical', 'tied', 'diag', 'full']

ll_spher = []
ll_tied = []
ll_diag = []
ll_full = []

ll_spher1 = []
ll_tied1 = []
ll_diag1 = []
ll_full1 = []

comp = []
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)

        gmm1 = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm1.fit(X_pca)

        bic.append(gmm.bic(X))
        bic1.append(gmm.bic(X))

        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        if bic1[-1] < lowest_bic1:
            lowest_bic1 = bic1[-1]
            best_gmm1 = gmm1
        if cv_type == 'spherical':
            ll_spher.append(gmm.score(X))
            ll_spher1.append(gmm1.score(X_pca))
            comp.append(n_components)
        if cv_type == 'tied':
            ll_tied.append(gmm.score(X))
            ll_tied1.append(gmm1.score(X_pca))
        if cv_type == 'diag':
            ll_diag.append(gmm.score(X))
            ll_diag1.append(gmm1.score(X_pca))
        if cv_type == 'full':
            ll_full.append(gmm.score(X))
            ll_full1.append(gmm1.score(X_pca))

ll_tied = np.array(ll_tied)
ll_spher = np.array(ll_spher)
ll_diag = np.array(ll_diag)
ll_full = np.array(ll_full)

ll_tied1 = np.array(ll_tied1)
ll_spher1 = np.array(ll_spher1)
ll_diag1 = np.array(ll_diag1)
ll_full1 = np.array(ll_tied1)


comp = np.array(comp)

fig, ax = plt.subplots()
plt.title("Adult Clustering EM")
plt.ylabel("Average Log-Likelihood")
plt.xlabel("n_components")
ax.plot(comp, ll_tied, label='tied Original', color='c')
ax.plot(comp, ll_spher, label='spher Original', color='m')
ax.plot(comp, ll_diag, label='diag Original', color='g')
ax.plot(comp, ll_full, label='full Original', color='y')


ax.plot(comp, ll_tied1, label='tied RP', color='c', linestyle=':')
ax.plot(comp, ll_spher1, label='spher RP', color='m', linestyle=':')
ax.plot(comp, ll_diag1, label='diag RP', color='g', linestyle=':')
ax.plot(comp, ll_full1, label='full RP', color='y', linestyle=':')

legend = ax.legend(shadow=True)
plt.show()