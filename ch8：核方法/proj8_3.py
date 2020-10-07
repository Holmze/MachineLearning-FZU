# from torchvision.datasets import CIFAR10

# CIFAR10(root='D:\Course\机器学习\MachineLearning-FZU\ch7：LR', download=True)
# # dset.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import clustering_performance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes,svm
import time

path = 'D:\Course\机器学习\MachineLearning-FZU\ch7：LR/cifar-10-batches-py/'


def unpickle(file):
    with open(file, 'rb') as fo:
        cifar = pickle.load(fo, encoding='bytes')
    return cifar


def test_LR(*data):
    X_train, X_test, y_train, y_test = data
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)
    # ACC = lr.score(X_test, y_test)
    # print('逻辑回归分类器')
    print('Logistic: %.4f' % lr.score(X_test, y_test))
    # print('Testing Score: %.4f' % ACC)
    return lr.score(X_test, y_test)


def test_GaussianNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()  # ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB','CategoricalNB']
    cls.fit(X_train, y_train)
    print('Bayes: %.4f' % cls.score(X_test, y_test))
    return cls.score(X_test, y_test)


def test_KNN(*data):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_sample = knn.predict(X_test)
    # print('KNN分类器')
    ACC = clustering_performance.clusteringMetrics(y_test, y_sample)
    print('KNN: %.4f' % ACC)
    return ACC

def test_SVM(*data):
    X_train, X_test, y_train, y_test = data
    clf = svm.SVC(C=2, kernel='sigmoid', gamma=10, decision_function_shape='ovr')
    clf.fit(X_train, y_train)
    y_predict_svm = clf.predict(X_test)
    print('SVM:','%.4f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
    return clf.score(X_test, y_test)

start_time = time.time()
test_data = unpickle(path + 'test_batch')
for i in range(1, 4):
    train_data = unpickle(path + 'data_batch_' + str(i))
    X_train, y_train = train_data[b'data'][0:1234], np.array(train_data[b'labels'][0:1234])
    X_test, y_test = test_data[b'data'][0:1234], np.array(test_data[b'labels'][0:1234])
    # print(X_train.size,X_test.size)
    # print()
    print('======','data_batch_' + str(i),'===========')
    test_KNN(X_train, X_test, y_train, y_test)
    test_GaussianNB(X_train, X_test, y_train, y_test)
    test_LR(X_train, X_test, y_train, y_test)
    test_SVM(X_train, X_test, y_train, y_test)
print(time.time()-start_time)