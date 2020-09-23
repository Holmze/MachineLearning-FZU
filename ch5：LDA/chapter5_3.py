# import cv2 as cv
# import os
# from PIL import Image

# path = 'D:\\Course\机器学习\\MachineLearning-FZU\\ch5：LDA\\LDA_dataset_17flowers'

# TrainFiles = os.listdir(path)  # 遍历每个子文件夹
# # 计算有几个文件(图片命名都是以 序号.jpg方式)
# Train_Number = len(TrainFiles)  # 子文件夹个数
# for k in range(0, Train_Number):
#     Trainneed = os.listdir(path + '\\' + TrainFiles[k])  # 遍历每个子文件夹里的每张图片
#     Trainneednumber = len(Trainneed)  # 每个子文件里的图片个数
#     print(Trainneednumber)
#     for i in range(0, Trainneednumber):
#         # image = cv.imread(path + '\\' + TrainFiles[k] + '\\' + Trainneed[i])
#         image = Image.open(path + '\\' + TrainFiles[k] + '\\' + Trainneed[i])
#         # print(path + '\\' + TrainFiles[k] + '\\' + Trainneed[i])
#         # print(image)
#         image.resize((200, 180))
#         if i == 1:
#             image.show()
        
#         # print(image)
#         # cv.imwrite(path + '\\' + TrainFiles[k] + '\\' + Trainneed[i], p)
# import cv2 as cv
# import os
# from PIL import Image

# path = 'D:\\Course\机器学习\\MachineLearning-FZU\\ch5：LDA\\LDA_dataset_17flowers'
# save_path = 'D:\\Course\机器学习\\MachineLearning-FZU\\ch5：LDA\\flowers'

# TrainImages = os.listdir(path)  # 遍历每个子文件夹
# # 计算有几个文件(图片命名都是以 序号.jpg方式)   TrainImages[0] = image_0001.jpg
# Train_Number = len(TrainImages)  # 所有图片个数

# # os.mkdir(save_path + '/flowers_new')

# for i in range(1, 18):  # 包头不包尾
#     os.mkdir(save_path + str(range(1, 18)[i - 1]))
# for k in range(0, Train_Number):
#     for i in range(1, 18):
#         image = Image.open(path + '/' + TrainImages[k])
#         if k//80+1 == range(1, 18)[i - 1]:
#             image.save(save_path + str(k//80+1) + '/' + TrainImages[k], image)
#             # print(TrainImages[k], k//80+1)
# import matplotlib.pyplot as plt
# import os
# import cv2 as cv
# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from PIL import Image
# from clustering_performance import clusteringMetrics1
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split

# path = 'D:\\Course\机器学习\\MachineLearning-FZU\\ch5：LDA\\LDA_dataset_17flowers'
# image_SIZE = 40
# IMAGE_COLUMN = 10  # 列
# IMAGE_ROW = 10  # 行
# to_image = Image.new('RGB', (IMAGE_COLUMN * image_SIZE, IMAGE_ROW * image_SIZE))


# def createDatabase(path):
#     # 查看路径下所有文件
#     TrainFiles = os.listdir(path)  # 遍历每个子文件夹
#     # 计算有几个文件(图片命名都是以 序号.jpg方式)
#     Train_Number = len(TrainFiles)  # 子文件夹个数
#     X_train = []
#     y_train = []
#     # 把所有图片转为1维并存入T中
#     for k in range(0, Train_Number):
#         Trainneed = os.listdir(path + '/' + TrainFiles[k])  # 遍历每个子文件夹里的每张图片
#         Trainneednumber = len(Trainneed)  # 每个子文件里的图片个数
#         for i in range(0, Trainneednumber):
#             img = Image.open(path + '/' + TrainFiles[k] + '/' + Trainneed[i]).resize((image_SIZE, image_SIZE), Image.ANTIALIAS)
#             to_image.paste(img, (i * image_SIZE, k * image_SIZE))  # 把读出来的图贴到figure上
#             image = Image.open(path + '/' + TrainFiles[k] + '/' + Trainneed[i])  # 数据类型转换
#             # image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # RGB变成灰度图
#             imzge = image.resize((666*500,1),1)
#             X_train.append(image)
#             y_train.append(k)
#             if i==1:
#                 print(image)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     return X_train, y_train


# X_train, y_train = createDatabase(path)
# # print(X_train.shape)
# X_train = X_train.reshape(X_train.shape[0], 180*200)
# # print(X_train)
# X_train, X_test, y_train, y_test = \
#     train_test_split(X_train, y_train, test_size=0.2, random_state=22)

# plt.figure()
# plt.imshow(to_image)
# plt.show()

# A_PCA = []
# A_LDA = []
# for i in range(1, 11):
#     # PCA + KNN
#     pca = PCA(n_components=i).fit(X_train)  # pca模型训练
#     # 将输入数据投影到特征面正交基上
#     X_train_pca = pca.transform(X_train)
#     X_test_pca = pca.transform(X_test)
#     knn = KNeighborsClassifier()
#     knn.fit(X_train_pca, y_train)
#     y_sample = knn.predict(X_test_pca)
#     ACC_PCA = clusteringMetrics1(y_test, y_sample)
#     A_PCA.append(ACC_PCA)
#     # LDA + KNN
#     lda = LinearDiscriminantAnalysis(n_components=i).fit(X_train, y_train)  # lda模型训练
#     # 将输入数据投影到特征面正交基上
#     X_train_lda = lda.transform(X_train)
#     X_test_lda = lda.transform(X_test)
#     knn = KNeighborsClassifier()
#     knn.fit(X_train_lda, y_train)
#     y_sample = knn.predict(X_test_lda)
#     ACC_LDA = clusteringMetrics1(y_test, y_sample)
#     A_LDA.append(ACC_LDA)

# # 画柱状图
# fig, ax = plt.subplots()
# bar_width = 0.35
# opacity = 0.6  # 不透明度
# index = np.arange(10)
# ax.set_xticks(index + bar_width / 2)

# cylinder1 = ax.bar(index, A_PCA, bar_width, alpha=opacity, color='b', label='PCA')
# cylinder2 = ax.bar(index + bar_width, A_LDA, bar_width, alpha=opacity, color='g', label='LDA')

# label = []  # 横坐标标签
# for j in range(1, 11):
#     label.append(j)
# ax.set_xticklabels(label)

# plt.ylabel('ACC')
# plt.xlabel('Component')
# ax.legend()  # 图例标签

# plt.show()

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
import time

def load_Image_Into_Matrix(path):
    time_start = time.time()
    cnt = 0
    print(os.listdir(path))
    for img_filefolder in os.listdir(path):
        cnt += 1
        print(cnt)
        img_matrix = np.zeros((1,350*350*3))
        target_matrix = np.zeros((1,1))
        for img_file in os.listdir(path + img_filefolder):
            print("read",path + img_filefolder + "\\" + img_file,"at",time.time()-time_start)
            img = Image.open(path + img_filefolder + "\\" + img_file)
            # print(img.size[0],img.size[1])
            if (img.size[0]<img.size[1]):
                img = img.rotate(90)
            img = img.crop((100,100,450,450))
            img = np.asarray(img).reshape(1,350*350*3)
            img_matrix = np.vstack((img_matrix,img))
            target_matrix = np.vstack((target_matrix,np.asarray(cnt)))
        # break
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

img_matrix,target_matrix = load_Image_Into_Matrix('D:\\Course\机器学习\\MachineLearning-FZU\\ch5：LDA\\LDA_dataset_17flowers\\')
lda = LinearDiscriminantAnalysis(n_components=10)
# print(n_features, n_classes)
knn = KNeighborsClassifier(n_neighbors=10)
train_data,test_data,train_y,test_y = train_test_split(img_matrix,target_matrix,test_size=0.2)
data = lda.fit_transform(train_data,train_y)
test_data = lda.transform(test_data)
print(data,train_y)
knn.fit(data,train_y)
pre_y = knn.predict(test_data)
print(accuracy_score(test_y,pre_y))