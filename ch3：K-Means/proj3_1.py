import sklearn.datasets
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import metrics
from sklearn.cluster import KMeans

fig=plt.figure(1)  
x1,y1=sklearn.datasets.make_circles(n_samples=400,factor=0.5,noise=0.05)
# 生成圆圈形状的二维数据集.factor：里圈和外圈的距离之比；n_samples：总点数

plt.subplot(111)
plt.title('make_circles')

k_means = KMeans(n_clusters=2, random_state=5)
y_pred = k_means.fit_predict(x1)
C_c= k_means.cluster_centers_
plt.scatter(x1[:, 0], x1[:, 1], c=y_pred)
plt.scatter(C_c[:,0],C_c[:,1],c='b')

print("ACC = ",metrics.accuracy_score(y1,y_pred))
print("MNI = ",metrics.normalized_mutual_info_score(y1,y_pred))
print("ARI = ",metrics.adjusted_rand_score(y1,y_pred))

plt.show()