from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import PIL.Image as Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from clustering_performance import clusteringMetrics
import random
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGES_PATH_DIR = os.getcwd() + r"\\face_images\\"
IMAGES_FORMAT = ['.jpg', '.py']
IMAGE_SIZE = (200,180)
IMAGE_ROW = 10
IMAGE_COLUMN = 20
IMAGE_SAVE_PATH = 'Image_cluster_.jpg'

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def loadData(FileName):
    im = Image.open(FileName)
    im = im.convert('RGB')
    # im = im.convert('gray')
    return im

data_show = plt.figure(1)

image_names = []
image_vector = []
real_label = sorted([_ for _ in range(0, 10)]*20)

to_image = Image.new('RGB', (IMAGE_SIZE[1]*IMAGE_COLUMN, IMAGE_SIZE[0]*IMAGE_ROW))
image_dir_names = [name for name in os.listdir(IMAGES_PATH_DIR) if os.path.splitext(name)[1] not in IMAGES_FORMAT ]

# print(image_dir_names)

for x, dir_name in enumerate(image_dir_names):
    new_addr = IMAGES_PATH_DIR + dir_name + '\\'
    image_names = [name for name in os.listdir(new_addr) if os.path.splitext(name)[1] in IMAGES_FORMAT ]
    # print(image_names)
        
    for y, _PATH in enumerate(image_names):
        IMAGES_PATH = new_addr + _PATH
        # print(IMAGES_PATH)
        im = loadData(IMAGES_PATH)
        image_vector.append(list(im.getdata()))

image_vector = np.array(image_vector).ravel().reshape((200, -1))

# 读取完毕

k=15
Pca = PCA(n_components=k)
Pca.fit(image_vector)
new_image_vector = Pca.transform(image_vector)

# print(normalization(np.mat(image_vector[10])*(np.mat(Pca.components_).T)))
# print(normalization(new_image_vector[10]))

for i, new_im in enumerate(Pca.components_):
    mini, maxi = min(new_im), max(new_im)

    new_im = np.array([(i-mini)/(maxi-mini) for i in new_im])
    new_im = new_im.reshape(200, 180, 3)

    # print(new_im)
    plt.subplot(1, k, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(new_im,cmap=plt.cm.gray)

plt.show()

"""
eigenface的python实现


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.decomposition import PCA

n_clusters = 10
image_resize_row = 180
image_resize_cols = 200
image_date_row = 1
image_date_cols = 10


# 后缀
suffix_list = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

# 图像类别(标签)
dataset_list = ["admars", "ahodki", "ajflem", "ajones", "ajsega",
                "anpage", "asamma", "asewil", "astefa", "drbost"]


def calculEigenfaces():
    # 数据集路径
    src_dir = ".//face_images//"
    # 获取图片以及对应类别（彩色）
    # imgs, labels = getGrayImgAndLabel(src_dir)
    imgs, labels = getColorImgAndLabel(src_dir)


    # 打印测试信息
    # print(len(labels))
    # for i in range(len(labels)):
    #     print(labels[i])


    # 将图片转换为数组(彩色)
    # arr = convertGrayImageToArrays(imgs)
    arr = convertColorImageToArrays(imgs)
    print("arr's shape : {}".format(arr.shape))
    # print(arr)




    # 先训练PCA模型,n_components为降到的维数
    pca = PCA(n_components=10)
    pca.fit(arr)

    # 返回测试集和训练集降维后的数据集
    arr_pca = pca.transform(arr)
    print("arr_pca的形状：", arr_pca.shape)

    for index in range(10):

        feature_img = convert_array_to_color_image(arr_pca[:,index:index+1], 200, 180)
        print("mean_arr's shape : {}".format(feature_img.shape))
        print("type(mean_img) : {}".format(type(feature_img)))
        # plt.imshow(mean_img)
        # plt.show()
        file_name = str(index) + '.png'
        cv2.imwrite(file_name,feature_img)

    # for index in range(10):
    #     # 先训练PCA模型,n_components为降到的维数
    #     pca = PCA(n_components=10)
    #     pca.fit(arr[:,index*20:index*20+20])
    #
    #     # 返回测试集和训练集降维后的数据集
    #     arr_pca = pca.transform(arr[:,index*20:index*20+20])
    #
    #     print("arr_pca的形状：", arr_pca.shape)
    #
    #     feature_img = convert_array_to_color_image(arr_pca, 200, 180)
    #     print("mean_arr's shape : {}".format(feature_img.shape))
    #     print("type(mean_img) : {}".format(type(feature_img)))
    #     # plt.imshow(mean_img)
    #     # plt.show()
    #     file_name = str(index) + '.png'
    #     cv2.imwrite(file_name,feature_img)


    # 计算均值图像
    mean_arr = compute_mean_array(arr)
    print("mean_arr's shape : {}".format(mean_arr.shape))
    # print(mean_arr)


    mean_img=convert_array_to_color_image(mean_arr, 200, 180)
    print ("mean_arr's shape : {}".format(mean_img.shape))
    print ("type(mean_img) : {}".format(type(mean_img)) )
    # plt.imshow(mean_img)
    # plt.show()
    cv2.imwrite("mean.png", mean_img)


    # 获取差值图像
    arr_diff = compute_diffs(arr, mean_arr)
    # 计算特征值以及特征向量
    eigenValues, eigenVectors = compute_eigenValues_eigenVectors(arr)
    print("eigenValues'shape : {}".format(eigenValues.shape))

    print("eigenVectors'shape : {}".format(eigenVectors.shape))


    # print eigenValues
    # 计算权重向量，此处假定使用特征值最大的前10个对应的特征向量作为基
    weights = compute_weights(arr_diff, eigenVectors[:, :10])
    print("weights.shape : {}".format(weights.shape))


    # print weights
    # 读取测试图像，此处使用训练库中的一张
    img_test = cv2.imread("gray/PEOPLE_GRAY_1/10.jpg", cv2.IMREAD_GRAYSCALE)
    arr_test = convertColorImageToArrays(img_test)
    diff_test = compute_diff(arr_test, mean_arr)
    wei = compute_weight(diff_test, eigenVectors[:, :3])
    print("test's weight : {}".format(wei))

    # 计算欧式距离
    weightValues = compute_euclidean_distances(weights, wei)
    print("weightValues.shape : {}".format(weightValues.shape))

    # 打印结果
    for i in range(len(weightValues)):
        print(weightValues[i], labels[i])





    for i in range(len(eigenValues)):
        img=convert_array_to_color_image(eigenVectors[:,i])
        cv2.imwrite(str(i)+".jpg" ,img)
        #break
        #print eigenValues[i]
    #endfor
    cv2.waitKey()
    print ("endl...")



def getGrayImgAndLabel(src_dir):
#    加载训练集
#    :param src_dir: 数据集路径
#    :return:
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取子文件夹名
    catelist = os.listdir(src_dir)
    # 遍历子文件夹
    for catename in catelist:
        # 设置子文件夹路径
        cate_dir = os.path.join(src_dir, catename)
        # 获取子文件名
        filelist = os.listdir(cate_dir)
        # 遍历所有文件名
        for filename in filelist:
            # 设置文件路径
            file_dir = os.path.join(cate_dir, filename)
            # 判断文件名是否为图片格式
            if not os.path.splitext(filename)[1] in suffix_list:
                print(file_dir, "is not an image")

                continue
            # endif
            # # 读取灰度图
            imgs.append(cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE))
            # 读取相应类别
            labels.append(catename)
    # endfor
    # endfor
    return imgs, labels


def getColorImgAndLabel_onedir(src_dir):
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取文件名
    filelist = os.listdir(src_dir)
    # 遍历所有文件名
    for filename in filelist:
        # 设置文件路径
        file_dir = os.path.join(src_dir, filename)
        # 判断文件名是否为图片格式
        if not os.path.splitext(filename)[1] in suffix_list:
            print(file_dir, "is not an image")

            continue
        # endif
        # # 读取彩色图像：三通道，unit8
        imgs.append(cv2.imread(file_dir))
        # 读取相应类别
        labels.append(filelist)
    return imgs, labels


def getColorImgAndLabel(src_dir):
    
#    加载训练集
#    :param src_dir: 数据集路径
#    :return:
    
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取子文件夹名
    catelist = os.listdir(src_dir)
    # 遍历子文件夹
    for catename in catelist:
        # 设置子文件夹路径
        cate_dir = os.path.join(src_dir, catename)
        # 获取子文件名
        filelist = os.listdir(cate_dir)
        # 遍历所有文件名
        for filename in filelist:
            # 设置文件路径
            file_dir = os.path.join(cate_dir, filename)
            # 判断文件名是否为图片格式
            if not os.path.splitext(filename)[1] in suffix_list:
                print(file_dir, "is not an image")

                continue
            # endif
            # # 读取彩色图像：三通道，unit8
            imgs.append(cv2.imread(file_dir))
            # 读取相应类别
            labels.append(catename)
    # endfor
    # endfor
    return imgs, labels


# end of getImgAndLabel

# 将图像（灰度）数据变为一列
def convertGrayImageToArray(img):
    img_arr = []
    height, width = img.shape[:2]
    # 遍历图像
    for i in range(height):
        img_arr.extend(img[i, :])
    # endfor
    return img_arr

# 将每个（灰度）图像变为一列
def convertGrayImageToArrays(imgs):
    # 初始化数组
    arr = []
    # 遍历每个图像
    for img in imgs:
        arr.append(convertGrayImageToArray(img))
    # endfor
    return np.array(arr).T


# 将图像（彩色）数据变为一列
def convertColorImageToArray(img):
    img_arr = []
    height, width, channel = img.shape[:3]

    # 遍历图像
    for h in range(height):
        for w in range(width):
            img_arr.extend(img[h][w][:])

    # print(np.array(img_arr).shape)

    return img_arr


# 将每个（彩色）图像变为一列
def convertColorImageToArrays(imgs):
    # 初始化数组
    arr = []
    # 遍历每个图像
    for img in imgs:
        arr.append(convertColorImageToArray(img))
    # endfor
    return np.array(arr).T


# 计算均值数组
def compute_mean_array(arr):
    # 获取维数(行数),图像数(列数)
    dimens, nums = arr.shape[:2]
    # 新建列表
    mean_arr = []
    # 遍历维数
    for i in range(dimens):
        # 求和每个图像在该字段的值并平均
        aver = int(sum(arr[i, :]) / float(nums))
        mean_arr.append(aver)
    # endfor
    return np.array(mean_arr)


# end of compute_mean_array


# 将数组转换为对应图像
def convert_array_to_image(arr, height=256, width=256):
    img =[]
    for i in range(height):
        img.append(arr[i * width:i * width + width])
    # endfor
    return np.array(img)


# 将数组转换为对应图像
def convert_array_to_color_image(arr, height=256, width=256):
    img = arr.reshape(200, 180, 3)
    # endfor
    return np.array(img)



# 计算图像和平均图像之间的差值
def compute_diff(arr, mean_arr):
    return arr - mean_arr


# end of compute_diff

# 计算每张图像和平均图像之间的差值
def compute_diffs(arr, mean_arr):
    diffs = []
    dimens, nums = arr.shape[:2]
    for i in range(nums):
        diffs.append(compute_diff(arr[:, i], mean_arr))
    # endfor
    return np.array(diffs).T


# end of compute_diffs

# 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列
# arr是预处理图像的矩阵，每一列对应一个减去均值图像之后的图像
def compute_eigenValues_eigenVectors(arr):
    arr = np.array(arr)
    # 计算arr'T * arr
    temp = np.dot(arr.T, arr)
    eigenValues, eigenVectors = np.linalg.eig(temp)
    # 将数值从大到小排序
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    # 特征向量按列排
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, np.dot(arr, eigenVectors)


# end of compute_eigenValues_eigenVectors

# 计算图像在基变换后的坐标(权重)
def compute_weight(img, vec):
    return np.dot(img, vec)


# end of compute_weight

# 计算图像权重
def compute_weights(imgs, vec):
    dimens, nums = imgs.shape[:2]
    weights = []
    for i in range(nums):
        weights.append(compute_weight(imgs[:, i], vec))
    return np.array(weights)


# end of compute_weights

# 计算两个权重之间的欧式距离
def compute_euclidean_distance(wei1, wei2):
    # 判断两个向量的长度是否相等
    if not len(wei1) == len(wei2):
        print('长度不相等')

        os._exit(1)
    # endif
    sqDiffVector = wei1 - wei2
    sqDiffVector = sqDiffVector ** 2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances ** 0.5
    return distance


# end of compute_euclidean_distance

# 计算待测图像与图像库中各图像权重的欧式距离
def compute_euclidean_distances(wei, wei_test):
    weightValues = []
    nums = wei.shape
    print(nums)

    for i in range(nums[0]):
        weightValues.append(compute_euclidean_distance(wei[i], wei_test))
    # endfor
    return np.array(weightValues)


# end of compute_euclidean_distances



# def image_compose_clusterdata(clusterdata):
#     
#     使用PIL.Image将聚类后的图像拼接成一张大图
#     :param clusterdata: 数据格式：List
#     :return:
#     
#     # list转为array
#     print("用于聚类的数据类型：", type(clusterdata),"数据维度：",clusterdata.shape)     # 打印数据维度与类型
#
#
#     # 创建一个新图 (mode,(width,height))
#     to_image = Image.new('RGB', (image_date_cols * image_resize_cols , image_date_row * image_resize_row))
#     print("to_image的大小",to_image.size)
#
#     # 遍历数组，将所有数据格式从array转为Image
#     for row in range(len(rawdata)-1):
#         for cols in range(len(rawdata[row])):
#             from_image = rawdata[row][cols].reshape(image_resize_cols, image_resize_row, 3)
#             from_image = Image.fromarray(cv.cvtColor(from_image,cv.COLOR_BGR2RGB))
#             to_image.paste(from_image, (cols * image_resize_cols, row* image_resize_row))
#
#
#     plt.imshow(to_image)
#     plt.title("ACC=%f, ARI=%f, NMI=%f"%(ACC, ARI, NMI ))
#     plt.show()
"""
