import sklearn.datasets
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import metrics
from sklearn.cluster import KMeans

fig=plt.figure(1)  
x1,y1=sklearn.datasets.make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.4])

plt.subplot(111)


k_means = KMeans(n_clusters=4, random_state=5)
y_pred = k_means.fit_predict(x1)
C_c= k_means.cluster_centers_
plt.scatter(x1[:, 0], x1[:, 1], c=y_pred)
plt.scatter(C_c[:,0],C_c[:,1],c='b')

print("ACC = ",metrics.accuracy_score(y1,y_pred))
print("MNI = ",metrics.normalized_mutual_info_score(y1,y_pred))
print("ARI = ",metrics.adjusted_rand_score(y1,y_pred))

plt.show()