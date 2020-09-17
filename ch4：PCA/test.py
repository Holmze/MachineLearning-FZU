from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
import pandas as pd
from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

img_List = []
target_List = []
cnt = 0
path = "D:\\Course\\机器学习\\MachineLearning-FZU\\ch3：K-Means\\project3_face_images\\face_images/"

# picture size : (200,180,3)
def cluster_acc(y_true, y_pred):

    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def presentation():
    def show_image(imgs,num_rows,num_cols,scale=2):
        figsize = (num_cols*scale,num_rows*scale)
        _,axes = plt.subplots(num_rows,num_cols,figsize=figsize)
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i][j].imshow(imgs[i*num_cols+j])
                axes[i][j].axes.get_xaxis().set_visible(False)
                axes[i][j].axes.get_yaxis().set_visible(False)
        return axes
    show_image([img_List[i][j] for i in range(len(img_List)) for j in range(len(img_List[0]))], 10, 20)
    plt.show()

def data_packaging(train,target):
    data = np.hstack([train,target])
    data = pd.DataFrame(data)
    # because Kmeans is sensitive to the order of data input
    data = shuffle(data)
    train_data = data.iloc[:,0:data.shape[1]-1]
    target_data = data.iloc[:,data.shape[1]-1]
    return train_data,target_data

for img_filefolder in os.listdir(path):
    cnt += 1
    sub_img_list = []
    sub_target_list = []
    for img_file in os.listdir(path+img_filefolder):
        img = Image.open(path+img_filefolder+"/"+img_file)
        img = np.asarray(img)
        # img = img.reshape((1,-1))
        sub_img_list.append(img)
        sub_target_list.append(np.array(cnt))
    img_List.append(sub_img_list)
    target_List.append(sub_target_list)

img_matrix = img_List[0][0]
target_vector = target_List[0][0]
for i in range(len(img_List)):
    for j in range(len(img_List[0])):
        img_matrix = np.vstack([img_matrix,img_List[i][j]])
        target_vector = np.vstack([target_vector,target_List[i][j]])
img_matrix = np.delete(img_matrix,0,0)
target_vector = np.delete(target_vector,0,0)
presentation()

train_data,target_data = data_packaging(img_matrix,target_vector)
train_data = train_data.values
target_data = target_data.values
all_list = []
acc_list = []
nmi_list = []
ari_list = []
for num in range(1,9):
    pca = PCA(n_components=num)
    data = pca.fit_transform(train_data)
    KM = KMeans(n_clusters=10)
    KM.fit(data)
    pre_y = KM.predict(data)
    acc_list.append(cluster_acc(pre_y,target_data))
    nmi_list.append(normalized_mutual_info_score(target_data,pre_y))
    ari_list.append(adjusted_rand_score(target_data, pre_y))
    print("ACC = ",cluster_acc(pre_y,target_data),"NMI = ",normalized_mutual_info_score(target_data,pre_y),"ARI = ",adjusted_rand_score(target_data,pre_y))

all_list.append(acc_list)
all_list.append(nmi_list)
all_list.append(ari_list)
all_list = np.array(all_list)
all_list = all_list.T
data = pd.DataFrame(all_list)
data.columns = ["ACC","NMI","ARI"]
data.plot(kind = 'bar')
plt.show()
