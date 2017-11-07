from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt

data = "C:\\Users\\nikhi\\Documents\\MachineLearning7641\\Projects\\Project 3\\Adult\\Data\\Adult.csv"
dataset = np.loadtxt(data, delimiter=",")
X = dataset[:,0:107]
y = dataset[:,108]
print(X.shape)

train_score = []
test_score = []
dimensions = []
for state in range(0,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    clf = ExtraTreesClassifier(random_state=state)
    clf = clf.fit(X_train, y_train)
    # print(clf.feature_importances_)
    train_score.append(clf.score(X_test, y_test))
    # print(clf.score(X_test, y_test))
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    dimensions.append(X_new.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0, test_size=0.3)
    clf = clf.fit(X_train, y_train)
    test_score.append(clf.score(X_test, y_test))
    print(clf.score(X_test, y_test))

print(np.argmax(np.array(test_score)))
print(train_score[np.argmax(np.array(test_score))])
print(test_score[np.argmax(np.array(test_score))])

figure, ax = plt.subplots()
ax.plot(np.array(range(0,11)), np.array(train_score), label='test before')
ax.plot(np.array(range(0,11)), np.array(test_score), label='test after')
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.title("Adult Gini")

plt.axvline(0, linestyle='--', label=str(dimensions[0])+' dim', color='b')
plt.axvline(1, linestyle='-.', label=str(dimensions[1])+' dim', color='g')
plt.axvline(2, linestyle=':', label=str(dimensions[2])+' dim', color='r')
plt.axvline(3, linestyle='-', label=str(dimensions[3])+' dim', color='c')
plt.axvline(4, linestyle='--', label=str(dimensions[4])+' dim', color='m')
plt.axvline(5, linestyle='-.', label=str(dimensions[5])+' dim', color='y')
plt.axvline(6, linestyle=':', label=str(dimensions[6])+' dim', color='k')
plt.axvline(7, linestyle='-', label=str(dimensions[7])+' dim', color='g')
plt.axvline(8, linestyle='--', label=str(dimensions[8])+' dim', color='r')
plt.axvline(9, linestyle='-.', label=str(dimensions[9])+' dim', color='c')
plt.axvline(10, linestyle=':', label=str(dimensions[10])+' dim', color='m')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# legend = ax.legend(shadow=True)
plt.show()