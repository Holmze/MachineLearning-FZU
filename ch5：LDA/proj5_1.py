# from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# digits = load_digits()
# X = digits.data
# labels = digits.target
# print(X)
# print(X.shape)


# # for i in range(100):
# #     plt.subplot(10,10,i+1)
# #     plt.imshow(X[i].reshape(8,8))
# #     plt.xticks([])
# #     plt.yticks([])
# # plt.show()

# X_train, X_test, labels_train, lables_test = train_test_split(X, labels, test_size=0.2, random_state=22)
# lda = PCA(n_components=2)
# lda.fit(X_train)
# x_train_re=lda.transform(X_train)    #对于训练数据和测试数据进行降维到二维数据
# x_test_re=lda.transform(X_test)
# print(x_train_re)
# knn1=KNeighborsClassifier()
# knn1.fit(x_train_re)             #再对降维到的二维数据进行KNN算法的训练和测试准确度
# print(knn1.score(x_test_re))

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from clustering_performance import clusteringMetrics1  # 这里修改一下老师代码的返回值，只返回要用到的ACC

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=22)

# 训练样本数1797  特征数64=8×8
# plt.gray()
# plt.matshow(digits.images[0])  # 显示0的图
# plt.show()

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 绘制数字：每张图像8*8像素点
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])  # 给空列表为了把坐标值去了
    ax.imshow(digits.images[i])  # interpolation='nearest'设置像素点边界模糊程度  cmap=plt.cm.binary设置颜色类型
    # 用目标值标记图像
    ax.text(0, 7, str(y_train[i]))
# fig.show()

A_PCA = []
A_LDA = []
for i in range(1, 10):
    # PCA + KNN
    pca = PCA(n_components=i).fit(X_train)  # pca模型训练
    # 将输入数据投影到特征面正交基上
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_pca, y_train)
    y_sample = knn.predict(X_test_pca)
    ACC_PCA = clusteringMetrics1(y_test, y_sample)
    A_PCA.append(ACC_PCA)
    # LDA + KNN
    print(i)
    lda = LinearDiscriminantAnalysis(n_components=i).fit(X_train, y_train)  # lda模型训练 记得加上y_train训练集的标签
    # 将输入数据投影到特征面正交基上
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_lda, y_train)
    y_sample = knn.predict(X_test_lda)
    ACC_LDA = clusteringMetrics1(y_test, y_sample)
    A_LDA.append(ACC_LDA)

# 画柱状图
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.6  # 不透明度
# print(type(A_LDA))
A_LDA = np.array(A_LDA)
# print(A_LDA,type(A_LDA))
A_PCA = np.array(A_PCA)
A_PCA_nparray = []
for i in range(len(A_PCA)):
    A_PCA_nparray.append(A_PCA[i][1])
A_PCA_nparray = np.array(A_PCA_nparray)
A_LDA_nparray = []
for i in range(len(A_LDA)):
    A_LDA_nparray.append(A_LDA[i][1])
A_LDA_nparray = np.array(A_LDA_nparray)
index = np.arange(len(A_PCA))
ax.set_xticks(index + bar_width / 2)
print("index:",len(index),"A_PCA:",A_PCA[0][1],"A_LDA:",len(A_LDA))

cylinder1 = ax.bar(index, A_PCA_nparray, bar_width, alpha=opacity, color='r', label='PCA')
cylinder2 = ax.bar(index + bar_width, A_LDA_nparray, bar_width, alpha=opacity, color='b', label='LDA')

label = []  # 横坐标标签
for j in range(1, 10):
    label.append(j)
ax.set_xticklabels(label)
ax.legend()  # 图例标签
plt.show()