from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import random

fig = plt.figure(1)
x1, y1 = make_circles(n_samples=400, factor=0.5, noise=0.1)     # 生成二维数组
# n_samples：生成样本数，内外平分   noise：异常点的比例   factor：内外圆之间的比例因子 ∈(0,1)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据

# # 模型训练
k = 15
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(x1, y1);
# # 进行预测
x2 = ((random.random())-0.5)*1.6
y2 = ((random.random())-0.5)*1.6
X_sample = np.array([[x2, y2]])
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(x1, return_distance=False)
##输出配置
plt.subplot(121)     # 一行两列 当前为1(第一行第一列)
plt.title('make_circles 1')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1,s=10,cmap='viridis')
plt.scatter(x2, y2, marker='*', s=10,c='b')
# 第一个参数横坐标 第二个参数纵坐标 marker为标志图案 c为颜色(可以是二维行数组)
# x1[:,0]即取x坐标 x1[:,1]即取y坐标

plt.subplot(122)
plt.title('make_circles 2')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1,s=10,cmap='viridis')
plt.scatter(x2, y2, marker='*', c='c',s=20,cmap='viridis')
for i in neighbors[0]:
    plt.plot([x1[i][0], X_sample[0][0]], [x1[i][1], X_sample[0][1]], '-', linewidth=0.6, c='b')
    plt.scatter([x1[i][0], X_sample[0][0]], [x1[i][1], X_sample[0][1]], marker='o', s=9,c='r')

plt.show()
