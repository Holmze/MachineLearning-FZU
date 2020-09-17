import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
# import general_fun
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


def load_Image_Into_Matrix(path):
    cnt = 0
    for img_filefolder in os.listdir(path):
        cnt += 1
        img_matrix = np.zeros((1,350*350*3))
        target_matrix = np.zeros((1,1))
        for img_file in os.listdir(path + img_filefolder):
            img = Image.open(path + img_filefolder + "/" + img_file)
            if (img.size[0]<img.size[1]):
                img = img.rotate(90)
            img = img.crop((100,100,450,450))
            img = np.asarray(img).reshape(1,350*350*3)
            img_matrix = np.vstack((img_matrix,img))
            target_matrix = np.vstack((target_matrix,np.asarray(cnt)))
        break
    img_matrix = np.delete(img_matrix,0,0)
    target_matrix = np.delete(target_matrix,0,0)
    print("complete!")
    return img_matrix,target_matrix

# img_matrix,target_matrix = load_Image_Into_Matrix("./17flowers/")
# lda = LinearDiscriminantAnalysis(n_components=17)
# knn = KNeighborsClassifier(n_neighbors=17)
# train_data,test_data,train_y,test_y = train_test_split(img_matrix,target_matrix,test_size=0.2)
# data = lda.fit_transform(train_data,train_y)
# test_data = lda.transform(test_data)
# knn.fit(data,train_y)
# pre_y = knn.predict(test_data)
# print(accuracy_score(test_y,pre_y))

img_matrix,target_matrix = load_Image_Into_Matrix('D:\Course\机器学习\MachineLearning-FZU\ch5：LDA\LDA_dataset_17flowers/17flowers/')
lda = LinearDiscriminantAnalysis(n_components=10)
knn = KNeighborsClassifier(n_neighbors=10)
train_data,test_data,train_y,test_y = train_test_split(img_matrix,target_matrix,test_size=0.2)
data = lda.fit_transform(train_data,train_y)
test_data = lda.transform(test_data)
knn.fit(data,train_y)
pre_y = knn.predict(test_data)
print(accuracy_score(test_y,pre_y))
