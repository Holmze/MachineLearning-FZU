from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 老师给的库
from clustering_performance import clusteringMetrics
import myModule.EmailFeatureGeneration as Email
import time

strat_time = time.time()
X, Y = Email.Text2Vector()
tst_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tst_size, random_state=22)
# print("X_train.shape =", X_train.shape)
# print("X_test.shape =", X_test.shape)
print("load data finish at",time.time()-strat_time,"s")
print("test size:",tst_size*100,"%","train size:",(1-tst_size)*100,"%")
# 朴素贝叶斯
clf = GaussianNB()
clf.fit(X_train, y_train)
print("finish Bayes fit at",time.time()-strat_time,"s")
y_sample_bayes = clf.predict(X_test)
Bayes_ACC = clusteringMetrics(y_test, y_sample_bayes)
print("Bayes Accuracy =", Bayes_ACC,"at",time.time()-strat_time,"s")

fig = plt.figure()
plt.subplot(121)
plt.title('Bayes heatmap')
confusion = confusion_matrix(y_sample_bayes, y_test)
confusion = confusion/X_test.shape[0]
# print(confusion)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.3g')
plt.xlabel('Predicted label')
plt.ylabel('True label')

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("finish KNN fit at",time.time()-strat_time,"s")
y_sample_knn = knn.predict(X_test)
KNN_ACC = clusteringMetrics(y_test, y_sample_knn)
print("KNN Accuracy =", KNN_ACC,"at",time.time()-strat_time,"s")

plt.subplot(122)
plt.title('KNN heatmap')
confusion = confusion_matrix(y_sample_knn, y_test)
confusion = confusion/X_test.shape[0]
sns.heatmap(confusion, annot=True, cmap='RdBu_r', fmt='.3g')
plt.xlabel('Predicted label')


plt.show()