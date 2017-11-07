import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import tree

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]

# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target

# Plot the PCA spectrum
pca.fit(X)
# print(pca.explained_variance_ratio_)
explained_variance_percentage = np.array(pca.explained_variance_ratio_)
explained_variance_percentage = [x * 100 for x in explained_variance_percentage]
explained_variance_percentage = np.cumsum(explained_variance_percentage)
print(explained_variance_percentage)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
# yyaxis left
# plt.plot(pca.explained_variance_ratio_, linewidth=2, label='explained_varia')
plt.plot(explained_variance_percentage, label='cumulative variance percentage')
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('Cumulative explained_variance % ')
plt.title("Adult PCA")
plt.show()

# Prediction
# n_components = range(1,11)
# Cs = np.logspace(-4, 4, 3)
#
# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# estimator = GridSearchCV(pipe,
#                          dict(pca__n_components=n_components,
#                               logistic__C=Cs))
# estimator.fit(X, y)
# print(estimator.best_estimator_.named_steps['pca'].n_components)
# # print(estimator.cv_results_)
# plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#             linestyle=':', label='n_components chosen')
# plt.legend(prop=dict(size=12))
