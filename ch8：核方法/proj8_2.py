# this code should run without conda python

from sklearn.datasets import load_digits
from sklearn import naive_bayes,svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from clustering_performance import cluster_acc
import clustering_performance
import numpy as np
import os
import cv2 as cv
import time
from PIL import Image


# KNN分类器
def test_KNN(*data):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    # nsamples, nx, ny = X_train.shape
    # d2_train_dataset = X_train.reshape((nsamples,nx*ny))
    # print(X_train, y_train)
    knn.fit(X_train, y_train)
    # print("fit over")
    y_pre = knn.predict(X_test)
    # print()
    ACC = cluster_acc(y_test, y_pre)
    # ACC = clusteringMetrics(y_test, y_pre)
    print('KNN分类器:','%.4f' % ACC)
    return ACC


# 高斯贝叶斯分类器
def test_GaussianNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()  # ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB','CategoricalNB']
    cls.fit(X_train, y_train)
    # print()
    # print('贝叶斯分类器')
    print('高斯贝叶斯分类器:','%.4f' % cls.score(X_test, y_test))
    return cls.score(X_test, y_test)

def test_Logistic(*data):
    X_train, X_test, y_train, y_test = data
    clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf.fit(X_train, y_train)
    print('LogisticRegression分类器:','%.4f' % clf.score(X_test, y_test))
    return clf.score(X_test, y_test)

def test_SVM(*data):
    X_train, X_test, y_train, y_test = data
    clf = svm.SVC(C=2, kernel='linear', gamma=10, decision_function_shape='ovr')
    clf.fit(X_train, y_train)
    y_predict_svm = clf.predict(X_test)
    print('SVM分类器:','%.4f' % clustering_performance.clusteringMetrics(y_test, y_predict_svm))
    return clf.score(X_test, y_test)

path_face = 'D:\Course\机器学习\MachineLearning-FZU\ch3：K-Means\project3_face_images\\face_images'
path_flower = 'D:\Course\机器学习\MachineLearning-FZU\ch5：LDA\LDA_dataset_17flowers'


# 读取Face image
def createDatabase(path):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)  # 遍历每个子文件夹
    # 计算有几个文件(图片命名都是以 序号.jpg方式)
    Train_Number = len(TrainFiles)  # 子文件夹个数
    print(Train_Number)
    X_train = []
    y_train = []
    # 把所有图片转为1维并存入X_train中
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path + '/' + TrainFiles[k])  # 遍历每个子文件夹里的每张图片
        Trainneednumber = len(Trainneed)  # 每个子文件里的图片个数
        for i in range(0, Trainneednumber):
            print(path + '/' + TrainFiles[k] + '/' + Trainneed[i])
            image = Image.open(path + '/' + TrainFiles[k] + '/' + Trainneed[i])  # 数据类型转换
            # print(path + '/' + TrainFiles[k] + '/' + Trainneed[i])
            image = image.convert('L') 
            # image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # RGB变成灰度图
            if (image.size[0]<image.size[1]):
                image = image.rotate(90)
            image = image.crop((100,100,450,450))
            # image = np.asarray(image)
            image = np.asarray(image).reshape(1,350*350)
            # print(image.size)
            # img_matrix = np.vstack((img_matrix,img))
            X_train.append(image)
            y_train.append(k)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


X_train_flower, y_train_flower = createDatabase(path_flower)
X_train_flower = X_train_flower.reshape(X_train_flower.shape[0], 350*350)
X_train_flower, X_test_flower, y_train_flower, y_test_flower = train_test_split(X_train_flower, y_train_flower, test_size=0.5, random_state=15)

digits = load_digits()
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(digits.data, digits.target, test_size=0.5, random_state=15)

X_train_face, y_train_face = createDatabase(path_face)
X_train_face = X_train_face.reshape(X_train_face.shape[0], 350*350)
X_train_face, X_test_face, y_train_face, y_test_face = train_test_split(X_train_face, y_train_face, test_size=0.2, random_state=15)

start_time = time.time()
print('17flowers')
test_KNN(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
test_GaussianNB(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
test_Logistic(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
test_SVM(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
print("==========================")
print('Digits')
test_KNN(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
test_GaussianNB(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
test_Logistic(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
test_SVM(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
print("==========================")
print('Face images')
test_KNN(X_train_face, X_test_face, y_train_face, y_test_face)
test_GaussianNB(X_train_face, X_test_face, y_train_face, y_test_face)
test_Logistic(X_train_face, X_test_face, y_train_face, y_test_face)
test_SVM(X_train_face, X_test_face, y_train_face, y_test_face)
# print("Face finish at",time.time()-start_time)

# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])#训练集
# Y = np.array([1, 1, 1, 2, 2, 2])#每个点的类标签
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X, Y)#要先训练(调用fit方法)才能预测(调用predict方法)
# print("==Predict result by predict==")
# print(clf.predict([[-0.8, -1]]))#预测该点类别
# print("==Predict result by predict_proba==")
# print(clf.predict_proba([[-0.8, -1]]))
# print("==Predict result by predict_log_proba==")
# print(clf.predict_log_proba([[-0.8, -1]]))