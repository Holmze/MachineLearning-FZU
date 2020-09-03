import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data
y = iris.target
loo = LeaveOneOut()
loo.get_n_splits(X)

K = []
Accuracy = []
for k in range(1, 20):
    correct = 0
    knn = KNeighborsClassifier(k)
    for train, test in loo.split(X):
        #print("train size=",train.size,"test size=",test.size)
        knn.fit(X[train], y[train])
        y_sample = knn.predict(X[test])
        if y_sample == y[test]:
            correct += 1
    K.append(k)
    Accuracy.append(correct/len(X))
    print('K： ', k)
    print('Accuracy= ', correct/len(X))

plt.plot(K, Accuracy,'g.-')
plt.title("iris datasets")
plt.xlabel('Accuracy')
plt.ylabel('K')
plt.show()