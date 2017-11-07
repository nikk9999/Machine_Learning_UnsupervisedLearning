
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
# n_samples = 500

# Generate random sample, two components
np.random.seed(0)
# C = np.array([[0., -0.1], [1.7, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
# print(X)

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\WineKMeans\\winequality-white.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:10]
y = dataset[:,11]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 21)
cv_types = ['spherical', 'tied', 'diag', 'full']
# cv_types = ['diag']
# ll_spher = []
# ll_tied = []
# ll_diag = []
ll_full = []
# comp = []
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        # if cv_type == 'spherical':
        #     ll_spher.append(gmm.score(X))
        #     comp.append(n_components)
        # if cv_type == 'tied':
        #     ll_tied.append(gmm.score(X))
        # if cv_type == 'diag':
        #     ll_diag.append(gmm.score(X))
        # if cv_type == 'full':
        #     ll_full.append(gmm.score(X))

# ll_tied = np.array(ll_tied)
# ll_spher = np.array(ll_spher)
# ll_diag = np.array(ll_diag)
# ll_full = np.array(ll_full)

# comp = np.array(comp)

# fig, ax = plt.subplots()
# plt.title("Adult Clustering EM")
# plt.ylabel("Average Log-Likelihood")
# plt.xlabel("n_components")
# ax.plot(comp, ll_tied, label='tied')
# ax.plot(comp, ll_spher, label='spher')
# ax.plot(comp, ll_diag, label='diag')
# ax.plot(comp, ll_full, label='full')
# legend = ax.legend(shadow=True)
# plt.show()
bic = np.array(bic)

color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []
print(str(lowest_bic))
# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.set_ylabel('BIC Score')
spl.legend([b[0] for b in bars], cv_types)
plt.show()
# Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)
# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     # print("V")
#     # print(v)
#     # print("W")
#     # print(w)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)
#
# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()