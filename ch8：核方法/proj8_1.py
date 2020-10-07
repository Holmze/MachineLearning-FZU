from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import clustering_performance

fig = plt.figure()

X_train, y_train = make_circles(n_samples=400, factor=0.5, noise=0.1)
X_test, y_test = make_circles(n_samples=400, factor=0.5, noise=0.1)


# 原图
plt.subplot(331)
plt.title('Circle')
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

# KNN
plt.subplot(332)
plt.title('KNN')
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)
print('KNN: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_knn))

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_knn)

# NaiveBayes
plt.subplot(333)
plt.title('NaiveBayes')

bayes = naive_bayes.GaussianNB()
bayes.fit(X_train, y_train)
y_predict_bayes = bayes.predict(X_test)
print('NaiveBayes: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_bayes))

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_bayes)

# Logistic
plt.subplot(334)
plt.title('Logistic')

logistic = LogisticRegression(max_iter=500, solver='liblinear')
logistic.fit(X_train, y_train)
y_predict_logistic = logistic.predict(X_test)
print('Logistic: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_logistic))

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_logistic)

plt.subplot(336)
plt.title('linear SVM')

# (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
clf = svm.SVC(C=2, kernel='linear', gamma=10, decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_predict_svm = clf.predict(X_test)
print('linear SVM: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_svm)

plt.subplot(337)
plt.title('ploy SVM')

# (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
clf = svm.SVC(C=2, kernel='poly', gamma=10, decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_predict_svm = clf.predict(X_test)
print('ploy SVM: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_svm)

plt.subplot(338)
plt.title('rbf SVM')
# (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
clf = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_predict_svm = clf.predict(X_test)
print('rbf SVM: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_svm)

plt.subplot(339)
plt.title('sigmoid SVM')
# (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
clf = svm.SVC(C=2, kernel='sigmoid', gamma=10, decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_predict_svm = clf.predict(X_test)
print('sigmoid SVM: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_svm)

# plt.subplot(339)
# plt.title('precomputed SVM')
# # (kernel) It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
# clf = svm.SVC(C=2, kernel='precomputed', gamma=10, decision_function_shape='ovr')
# clf.fit(X_train, y_train)
# y_predict_svm = clf.predict(X_test)
# print('precomputed SVM: %.3f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
# plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_predict_svm)

plt.show()
