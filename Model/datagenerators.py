import os
import SimpleITK as sitk
from PIL import  Image
import numpy as np
from torch.utils import data
from torchvision import transforms as T

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms=T.Compose([
            T.Resize(512), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            T.CenterCrop(512), # 512*512
            T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
            T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path).convert('L')
        label=0

        data = self.transforms(data)

        return data,label



class Dataset_3D(data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr