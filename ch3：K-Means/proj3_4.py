import numpy as np
import pylab as plt
import PIL.Image as image
from sklearn.cluster import KMeans

img = plt.imread('MachineLearning-FZU\ch3：K-Means\StoneHenge\StoneHenge.jpg')

img1 = img.reshape((img.shape[0]*img.shape[1], 3))

def k_means(k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img1)
    height = img.shape[0] 
    width = img.shape[1]
    pic_new = image.new("RGB", (width, height))
    print(width,",",height)
    center = np.zeros([k, 3])
    for i in range(k):
        for j in range(3):
            center[i, j] = kmeans.cluster_centers_[i, j]
    center = center.astype(np.int32)
    label = kmeans.labels_.reshape((height, width))
    for i in range(height):
        for j in range(width):
            pic_new.putpixel((j, i), tuple((center[label[i][j]])))
    pic_new.save("MachineLearning-FZU\ch3：K-Means\StoneHenge\StoneHenge_"+str(k)+"Means.jpg", "JPEG")

for i in range(2,10):
    k_means(i)