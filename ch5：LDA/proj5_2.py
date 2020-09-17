import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def bzh(data):  # 归一化
    x_max = np.max(data, axis=0)
    x_min = np.min(data, axis=0)
    xinx = (data-x_min)/(x_max-x_min)
    return xinx


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=22)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_train_pca = bzh(X_train_pca)
X_test_pca = pca.transform(X_test)
X_test_pca = bzh(X_test_pca)
knn = KNeighborsClassifier()
knn.fit(X_train_pca, y_train)
y_sample_pca = knn.predict(X_test_pca)

lda = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
X_train_lda = bzh(X_train_lda)
X_test_lda = lda.transform(X_test)
X_test_lda = bzh(X_test_lda)
knn = KNeighborsClassifier()
knn.fit(X_train_lda, y_train)
y_sample_lda = knn.predict(X_test_lda)

# 画图
figure = plt.figure()
ax1 = figure.add_subplot(1, 2, 1)
ax2 = figure.add_subplot(1, 2, 2)
for i in range(1437):
    # fontdict={'weight': 'bold', 'size': 9} 调整字体粗细
    ax1.text(X_train_pca[i, 0], X_train_pca[i, 1], str(y_train[i]), color=plt.cm.Set1(y_train[i]/10.))
    ax2.text(X_train_lda[i, 0], X_train_lda[i, 1], str(y_train[i]), color=plt.cm.Set1(y_train[i]/10.))
for j in range(360):
    ax1.text(X_test_pca[j, 0], X_test_pca[j, 1], str(y_sample_pca[j]))
    ax2.text(X_test_lda[j, 0], X_test_lda[j, 1], str(y_sample_lda[j]))

ax1.set_title("PCA")
ax2.set_title("LDA")

plt.show()
