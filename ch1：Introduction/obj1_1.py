###object1.1:sklearn.datasets.make_circle and sklearn.datasets.make_moons

import sklearn.datasets
import matplotlib.pyplot as plt  
import numpy as np  

fig=plt.figure(1)  
x1,y1=sklearn.datasets.make_circles(n_samples=400,factor=0.5,noise=0.05)
# 生成圆圈形状的二维数据集.factor：里圈和外圈的距离之比；n_samples：总点数

plt.subplot(121)
plt.title('make_circles')
plt.scatter(x1[:,0],x1[:,1],c=y1) #画图

plt.subplot(122)
x1,y1=sklearn.datasets.make_moons(n_samples=400,noise=0.05)
# 生成双月牙形状的二维数据集；n_samples：总点数

plt.title('make_moons')
plt.scatter(x1[:,0],x1[:,1],c=y1) #画图

plt.show() 