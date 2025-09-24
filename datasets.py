
import pandas as pd
import os
import cv2
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

## 数据提取
class SherdDataset:
    def __init__(self, csv_path, image_dir):
        self.info = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.available_images = set(os.listdir(image_dir))

    def get_image_path(self, image_id):
        return os.path.join(self.image_dir, f"{image_id}.jpg")

    def image_exists(self, image_id):
        return f"{image_id}.jpg" in self.available_images

    def load_image(self, image_id):
        path = self.get_image_path(image_id)
        if not os.path.exists(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        return img

    def iter_sherds(self):
        for _, row in self.info.iterrows():
            image_id = row['image_id']
            if self.image_exists(image_id):
                # sherd_id-文物, image_id-图片的exterior和interior, row-csv的信息
                yield row['sherd_id'], image_id, row
###微调
######################################################
class Dataset:
    def __init__(self, type_path, image_dir, label_path, transform):
        self.info = pd.read_csv(type_path)
        self.image_dir = image_dir
        self.available_images = set(os.listdir(image_dir))
        self.labels = pd.read_csv(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        #print(len(self.info),index)
        row = self.info.loc[index]
        image_id = row.loc['image_id']
        print(image_id)
        type_id = np.array(self.labels.loc[index])
        #print(image_id, type_id)
        if f"{image_id}.jpg" in self.available_images  and not pd.isnull(image_id):
            path = os.path.join(self.image_dir, f"{image_id}.jpg")
            #print(path)
            #if not os.path.exists(path):
                #return self.__getitem__(index + 1)
            img = cv2.imread(path)[200:800,200:800,:]
            img = self.transform(img)
            #if img is None:
                #return self.__getitem__(index + 1)
            return img, type_id
        else:
            return self.__getitem__(index + 1)

class Datasets:
    def __init__(self, type_path, image_dir, label_path, transform):
        self.info = pd.read_csv(type_path)
        self.image_dir = image_dir
        self.available_images = set(os.listdir(image_dir))
        self.labels = pd.read_csv(label_path)
        self.transform = transform
        self._len = len(self.info)

    def __getitem__(self, index):
        #index = index % self._len
        row = self.info.loc[index]
        image_id = row.loc['image_id']
        #print(image_id)
        type_id = np.array(self.labels.loc[index])
        #print(image_id, type_id)
        if f"{image_id}.jpg" in self.available_images:
            path = os.path.join(self.image_dir, f"{image_id}.jpg")
            img = cv2.imread(path)[200:800,200:800,:]
            img = self.transform(img)
            return img, type_id
        else:
            self.info = self.info.drop(self.info.index[index])
            self.info.reset_index(drop=True, inplace=True)
            self.labels = self.labels.drop(self.labels.index[index])
            self.labels.reset_index(drop=True, inplace=True)
            #print(len(self.info))
            self._len = len(self.info)
            return self.__getitem__(index)

    def __len__(self):
        return 440
################################################################
##预训练
class Datasetpre:
    def __init__(self, type_path, image_dir,transform):
        self.info = pd.read_csv(type_path)
        self.image_dir = image_dir
        self.available_images = set(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        #print(len(self.info),index)
        row = self.info.loc[index]
        image_id = row.loc['image_id']
        #ids = row.loc['image_id']
        #type_id = row.loc['type']
        #print(image_id)
        if f"{image_id}.jpg" in self.available_images:# and  not pd.isnull(type_id):
            path = os.path.join(self.image_dir, f"{image_id}.jpg")
            #print(path)
            img = cv2.imread(path)[200:800,200:800,:]
            ## 数据增强
            imgL = self.transform(img)
            imgR = self.transform(img)
            return imgL, imgR#, img, image_id
        else:
            return self.__getitem__(index + 1)
######################################################
def one_hot(y,num_class,label_index):
    one_h = np.zeros((len(y),num_class),dtype=int)
    for index, label in enumerate(y):
        j = label.split(",")
        for key in j:
            one_h[index,label_index[key.strip()]] = 1
    return one_h

if __name__ == '__main__':
    ##
    all_path = 'D:/note/h690/jd_sherds_info.csv'
    interior_path = 'D:/note/h690/interior.csv'
    image_dir = 'D:/note/h690/sherd_images'
    ## class
    all_class = 'D:/note/h690/all_jd_info.csv'
    all_label = 'D:/note/h690/all_label_info.csv'
    ## 内部
    ex_class = 'D:/note/h690/ex_jd_info.csv'
    ex_label = 'D:/note/h690/ex_label_info.csv'

    #info = pd.read_csv(csv_path)
    #info_drop = info.dropna()
    #info_drop.to_csv(out_path, index=False)

    labels = ['li','yan','zun (large mouthed)','guan','ding','gui','dou','pen','gang','wen','xiao guan','zeng','pan']
    label_index = {label: index for index, label in enumerate(labels)}
    #info = pd.read_csv(out_path, encoding='GBK')
    #target = info['type']
    #print(target.value_counts())
    ###one-hot编码
    #target = one_hot(target, len(labels), label_index)
    #target = pd.DataFrame(target)
    #print(target)
    #target.to_csv(label_path, index=False)
    '''
    info_type = pd.read_csv(all_class, encoding='GBK')
    info_label = pd.read_csv(all_label, encoding='GBK')
    print(info_type,info_label)
    mask = info_type['image_id'].astype(str).str.contains('exterior', na=False)
    ex_type_info = info_type[mask]  # 取出所有满足条件的行
    ex_label_info = info_label[mask]  # 取出所有满足条件的行
    print(ex_type_info, ex_label_info)
    # 保存
    ex_type_info.to_csv(ex_class, index=False)
    ex_label_info.to_csv(ex_label, index=False, header=False)'''

    ##
    transform = transforms.Compose(
        [transforms.ToPILImage(),# transforms.Resize(64),
         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机应用颜色抖动（概率为0.8）
         transforms.RandomGrayscale(p=0.2), transforms.ToTensor()])  # 图像转为灰度图像（概率为0.2）
    datasets = Datasets(ex_class, image_dir, ex_label, transform)
    dl_loder = DataLoader(datasets, batch_size = 1, shuffle = False, num_workers = 1)
    for i, (img, type_id) in enumerate(dl_loder, 1):
        print(i, img.shape, type_id)
    '''
        figs = plt.figure(dpi=120)
        ax = figs.add_subplot(1,3,1)
        ax.imshow(img.squeeze())
        ax1 = figs.add_subplot(1,3,2)
        ax1.imshow(imgR.squeeze().permute(1,2,0))
        ax2 = figs.add_subplot(1,3,3)
        ax2.imshow(imgL.squeeze().permute(1,2,0))
        plt.show()'''
