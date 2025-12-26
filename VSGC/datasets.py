import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
'''
glob用于查找符合特定模式的文件路径。
numpy用于数值计算。
os用于操作文件和目录。
scipy.io用于读取 .mat 文件。
torch:PyTorch 库，用于构建深度学习模型。
torch.utils.data.Dataset:PyTorch 中的数据集基类
'''

class trainset_loader(Dataset):
    def __init__(self):
        # self.file_path = 'input_' + dose
        # self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))
        self.files_A = sorted(glob.glob('/data/ssd/wyy/mouse_limited/train_90/*.mat'))
    def __getitem__(self, index):
        file_A = self.files_A[index]
        # file_B = file_A.replace(self.file_path,'label_single')
        # file_C = file_A.replace('input','projection')
        #file_B = file_A.replace('label','projection')
        #projdata = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_A)['imagef']
        #label_data2 = scio.loadmat(file_A)['imagef2']
        input_data = scio.loadmat(file_A)['imagelimit']
        #input_data2 = scio.loadmat(file_A)['imagelimit2']
        
        #projdata = torch.FloatTensor(projdata).unsqueeze_(0)
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        #label_data2 = torch.FloatTensor(label_data2).unsqueeze_(0)
        #input_data2 = torch.FloatTensor(input_data2).unsqueeze_(0)
        return input_data, label_data

    def __len__(self):
        return len(self.files_A)



class testset_loader(Dataset):
    def __init__(self):
        # 查找并排序所有测试数据文件路径，这些文件位于 ./test/label/ 目录下，扩展名为 .mat
        #获取阶段2数据集需要改变路径
        self.files_A = sorted(glob.glob('/data/ssd/wyy/mouse_limited/test_90/*.mat'))
    # 获取单个数据样本
    def __getitem__(self, index):
        file_A = self.files_A[index]
        # file_B = file_A.replace(self.file_path,'label_single')
        # file_C = file_A.replace('input','projection')
        #file_B = file_A.replace('label','projection')
        # 结果文件的名称，从 file_A 的最后 9 个字符生成,
        #res_name = 'D:/24\wyy\wyy\wyy\IRON_V1.0/rolling(IRON)\CAFM_Top-K_Distillation/result' + file_A[-9:]
        #res_name 是指标.mat文件# 不存了
        
        # projdata：投影数据
        projdata = scio.loadmat(file_A)['data']
        input_data = scio.loadmat(file_A)['imagelimit']
        # label_data：一标签数据
        label_data = scio.loadmat(file_A)['imagef']
        # label_data2：第二个标签数据
        label_data2 = scio.loadmat(file_A)['imagef2']
        input_data2 = scio.loadmat(file_A)['imagelimit2']
        
        projdata = torch.FloatTensor(projdata).unsqueeze_(0)
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        input_data2 = torch.FloatTensor(input_data2).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        label_data2 = torch.FloatTensor(label_data2).unsqueeze_(0)
        
        return input_data, label_data, input_data2, label_data2,projdata
    def __len__(self):
        return len(self.files_A)
