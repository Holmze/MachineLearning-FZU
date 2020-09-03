import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

fo = open('MachineLearning-FZU\ch2：KNN\cifar-10-python\cifar-10-python\cifar-10-batches-py\data_batch_1','rb')
train_data = pickle.load(fo,encoding='bytes')
fo.close()
fo = open('MachineLearning-FZU\ch2：KNN\cifar-10-python\cifar-10-python\cifar-10-batches-py\\test_batch','rb')
test_data = pickle.load(fo,encoding='bytes')
fo.close()
#print(train_data)
print("data size= ",test_data[b'data'].size/3072)

t0 = time.time()
knn = KNeighborsClassifier(10)#k=10
knn.fit(test_data[b'data'],np.array(train_data[b'labels']))
pred = knn.predict(test_data[b'data'][:5000])
print("test size: ",pred.size)
accuracy = np.sum(np.array(pred)==np.array(test_data[b'labels'][:5000]))/len(pred)
print("Accuracy= ",accuracy)
print("Time: ",(time.time()-t0),"Sec")