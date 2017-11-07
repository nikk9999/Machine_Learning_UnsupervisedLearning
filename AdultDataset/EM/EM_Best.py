import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)


np.random.seed(0)

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 15)
# cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = ['diag']
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

bic = np.array(bic)

color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []
print(str(lowest_bic))
# Plot the BIC scores
# spl = plt.subplot(2, 1, 1)

# colors = ['navy', 'turquoise', 'darkorange']
#
# def make_ellipses(gmm, ax):
#     for n, color in enumerate(colors):
#         if gmm.covariance_type == 'full':
#             covariances = gmm.covariances_[n][:2, :2]
#         elif gmm.covariance_type == 'tied':
#             covariances = gmm.covariances_[:2, :2]
#         elif gmm.covariance_type == 'diag':
#             covariances = np.diag(gmm.covariances_[n][:2])
#         elif gmm.covariance_type == 'spherical':
#             covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
#         v, w = np.linalg.eigh(covariances)
#         u = w[0] / np.linalg.norm(w[0])
#         angle = np.arctan2(u[1], u[0])
#         angle = 180 * angle / np.pi  # convert to degrees
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
#                                   180 + angle, color=color)
#         ell.set_clip_box(ax.bbox)
#         ell.set_alpha(0.5)
#         ax.add_artist(ell)
#

# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.set_ylabel('BIC Score')
# spl.legend([b[0] for b in bars], cv_types)
# plt.show()
# Plot the winner
# splot = plt.subplot(3, 3, 1)
fig, ax = plt.subplots()
Y_ = clf.predict(X)
# print("Y")
# print(Y_)
# print( clf.means_)
# print(clf.covariances_)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    print("Hi")
    v, w = linalg.eigh(np.diag(gmm.covariances_[i][:2]))
    if not np.any(Y_ == i):
        continue
    # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    plt.scatter(X[:, 0], X[:, 1], s=0.8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(.5)
    ax.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()