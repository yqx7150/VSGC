import argparse
import os
import re
import glob
import numpy as np
import scipy.io as sio
import cv2
# from vis_tools import Visualizer
# 这段代码的作用是指定哪些 GPU 设备对当前进程可见
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import modelcat

from datasets import trainset_loader
from datasets import testset_loader
from torch.utils.data import DataLoader
from hybrid_loss_founction import HybridLoss

from skimage import metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import time
torch.backends.cudnn.enabled = False


# import torch.distributed as dist

# dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=2)
# dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=1, world_size=2)


parser = argparse.ArgumentParser()
#--epochs 50 表示训练50个epochW
parser.add_argument("--epochs", type=int, default=34, help="number of epochs of training")
#A--batch_size 16 表示每个批次包含16个样本
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")#2
#Adam优化器的学习率
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=1)#40
#用于数据加载的CPU线程数
parser.add_argument("--n_cpu", type=int, default=10)
#A模型保存路径
parser.add_argument("--model_save_path", type=str, default="/home/ly/wyy/Ours_WTConv_edgeloss_weight/xiaoshu_result_90/model")
#--checkpoint_interval 5 表示每5个epoch保存一次检查点
parser.add_argument('--checkpoint_interval', type=int, default=1)
opt = parser.parse_args()
#cuda = True if torch.cuda.is_available() else False
cuda = True
# train_vis = Visualizer(env='training_magic')
imagesize=256

#关注区域可视化
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
    
        # 前向传播
        output1, output2 = self.model(input_tensor)
    
        # 对于图像重建任务，我们不需要类别，直接对输出计算梯度
        self.model.zero_grad() 
    
        # 方法1: 对整个输出图像计算梯度
        # 使用输出图像的平均值作为目标
        output1.mean().backward(retain_graph=True)
    
        # 全局平均池化梯度
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
    
        # 加权激活图
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
    
        # 生成热力图
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # 归一化
    
        return heatmap

# 使用示例
def visualize_attention(model, input_image, target_layer_name):
    """
    可视化模型关注区域
    
    Args:
        model: 训练好的模型
        input_image: 输入图像 (tensor格式)
        target_layer_name: 目标层名称，例如 'dnn.block8'
    """
    #打印模型层名看看测试一下
    #for name, layer in model.named_modules():
        #print(f"Layer: {name}, Type: {type(layer)}")

    # 获取目标层
    target_layer = dict(model.named_modules())[target_layer_name]
    
    # 创建Grad-CAM对象
    grad_cam = GradCAM(model, target_layer)
    
    # 生成热力图
    heatmap = grad_cam.generate_cam(input_image.unsqueeze(0))
    
    # 调整热力图大小以匹配原图
    heatmap = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[1]))
    
    # 将热力图转换为RGB
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    return heatmap_color
# 在你的模型中使用
# 假设你有一个训练好的模型和输入图像
# model = DDNET()  # 或你的完整模型
# input_tensor = torch.randn(1, 1, 256, 256)  # 示例输入

# 可视化特定层的关注区域
# heatmap = visualize_attention(model, input_tensor[0], 'dnn.block8')

class net():
    def __init__(self):
        '''
        创建模型实例 self.model,并设置其参数。
        定义损失函数为均方误差损失 self.loss = nn.MSELoss()。
        设置模型保存路径 self.path。
        加载训练数据集和测试数据集。
        初始化优化器 self.optimizer。
        检查是否已保存模型，若无则初始化权重。
        设置学习率调度器 self.scheduler。
        如果使用 GPU，则将模型移动到 GPU
        '''
        '''
        block_num：表示 IterBlock 模块的数量。
        **kwargs：包含其他配置选项，具体包括：
        views：视图数量。
        dets：检测数量。
        width：图像宽度。
        height：图像高度。
        dImg：图像深度。
        dDet：检测深度。
        Ang0：起始角度。
        dAng：角度步长。
        s2r：采样到重建的比例。
        d2r：检测到重建的比例。
        binshift：二进制位移。
        scale：缩放比例。
        scan_type：扫描类型
       
        '''

         # 创建一个 SummaryWriter 实例
        self.writer = SummaryWriter(log_dir='/home/ly/wyy/Ours_WTConv_edgeloss_weight/xiaoshu_result_90/lossfig')  # 创建模型实例
        self.model = modelcat.VSGC().cuda()
        '''
        无用的配置参数
        opt.n_block, views=180, dets=720, width=imagesize, height=imagesize, 
            dImg=0.009*2, dDet=0.0011,Ang0=0, dAng=2*3.14159/2200*7, s2r=5.3852, d2r=0, binshift=-0.0013,scale=3,scan_type=1
        '''
        self.model = torch.nn.DataParallel(self.model)
        #待修改损失函数
        #self.loss = nn.MSELoss()
        self.loss = HybridLoss()
        self.path = opt.model_save_path
        self.train_data = DataLoader(trainset_loader(),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        self.test_data = DataLoader(testset_loader(),
            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=opt.lr)
        self.start = 0
        self.epoch = opt.epochs
        self.check_saved_model()       
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size = 5, gamma=0.8)
        self.save_path = '/home/ly/wyy/Ours_WTConv_edgeloss_weight/xiaoshu_result_90'
        if cuda:
            self.model = self.model.cuda()
    #如果保存路径不存在，则创建目录并初始化模型权重。
    #如果存在保存的模型，则加载最新的模型权重，可以在原epoch上继续训练。30轮不够就再加，训练起点是最新的epoch
    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/catmodel_epoch_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'catmodel_epoch_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                self.model.load_state_dict(torch.load(
                    '%s/catmodel_epoch_%04d.pth' % (self.path, last_epoch)))

    #将图像像素值归一化到 [0, 255] 范围内
    
    def displaywin(self, img):
        img[img<0] = 0
        high=img.max()
        low=0
        img = (img - low)/(high - low) * 255
        return img
    #对卷积层和批量归一化层的权重进行初始化
    def initialize_weights(self):
        for module in self.model.modules():
            # if isinstance(module, modelACIDaddproj.prj_module):
            #     nn.init.normal_(module.weight, mean=0.05, std=0.01)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
       
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
        #计算模型参数总数
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        # 检查可用设备
        print("Available devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        '''
        进行多个 epoch 的训练，每个 epoch 包含多个 batch。
        在每个 batch 中，前向传播计算输出，计算损失，反向传播更新模型参数。
        使用学习率调度器调整学习率。
        根据设定的间隔保存模型
        '''
        for epoch in range(self.start, self.epoch):
            for batch_index, data in enumerate(self.train_data):
                input_data, label_data = data              
                if cuda:
                    input_data = input_data.cuda()
                    label_data = label_data.cuda()
                    #label_data2 = label_data2.cuda()
                    #input_data2 = input_data2.cuda()
                    #input_data2用在哪了？
                    

                self.optimizer.zero_grad()
                
                deep, deep2= self.model(input_data)
                total_loss1,  deep_loss, perceptual_loss, ssim_loss , edge_loss= self.loss(deep ,label_data)
                total_loss2,  deep_loss, perceptual_loss, ssim_loss , edge_loss= self.loss(deep2,label_data)
                loss=total_loss1+0.05*total_loss2

                
                loss.backward()

                # ==== 新增梯度裁剪代码 ====
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # ========================
                
                self.optimizer.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d]: [total_loss: %f] [deep_loss: %f] [perceptual_loss: %f] [ssim_loss: %f] [edge_loss: %f]"
                    % (epoch+1, self.epoch, batch_index+1, len(self.train_data),loss.item()
                    , deep_loss.item(), perceptual_loss.item(), ssim_loss.item(), edge_loss.item())
                )    
                # 将损失值写入 TensorBoard 日志
                global_step = epoch * len(self.train_data) + batch_index
                self.writer.add_scalar('Train/Total_Loss', loss.item(), global_step)   

            self.scheduler.step()#每轮训练更新学习率

            if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), '%s/catmodel_epoch_%04d.pth' % (self.path, epoch+1))
        # 关闭 TensorBoard writer
        self.writer.close()
            
    '''
    定义一个日志记录类 Logger，用于同时输出到控制台和文件。
    遍历测试数据集，对每个 batch 进行前向传播。
    计算预测结果与真实标签之间的损失、PSNR、SSIM 和 ERN。
    将结果保存为 .mat 文件。
    输出平均的 PSNR、SSIM 和 ERN
    '''

    def normalize_array(self, data):
        """
        对输入的 numpy 数组进行归一化处理，使其值范围在 0 到 1 之间

        :param data: 输入的 numpy 数组
        :return: 归一化后的数组
        """
        # 确保输入的是 numpy 数组
        # if not isinstance(data, np.ndarray):
            # raise ValueError("输入必须是 numpy 数组")

        # 归一化处理
        min_value = data.min()
        max_value = data.max()

        # 处理边界情况
        if max_value - min_value == 0:
            normalized_data = np.zeros_like(data)  # 如果所有值都相同，归一化后为全零
        else:
            normalized_data = (data - min_value) / (max_value - min_value)

        return normalized_data


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x = self.normalize_array(x)
        y= self.normalize_array(y)
        pred =  self.normalize_array(pred)
        x = np.squeeze(x)
        y = np.squeeze(y)
        pred = np.squeeze(pred)
        #x, y, pred = x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[0].set_title('limit-angle 90', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}".format(original_result[0],
                                                             original_result[1],
                                                             ), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[1].set_title('CAT_result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}".format(pred_result[0],
                                                             pred_result[1],
                                                             ), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[2].set_title('Original', fontsize=30)
        # 还差一个图片保存路径
        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()
        print("completed  {}, psnr is {}, ssim is {}".format(fig_name, pred_result[0], pred_result[1]))
     
    def save_one_fig(self, pred, figname):
        pred =  self.normalize_array(pred)
        pred = np.squeeze(pred)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(pred, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.set_position([0, 0, 1, 1])  # 设置图像的边界为满画布，去除白边
        f.savefig(os.path.join(self.save_path, 'fig', '{}.png'.format(figname)))
        plt.close()

    def save_one_gt(self, y, figname):
        y =  self.normalize_array(y)
        y = np.squeeze(y)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(y, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.set_position([0, 0, 1, 1])  # 设置图像的边界为满画布，去除白边
        f.savefig(os.path.join(self.save_path, 'fig', '{}_gt.png'.format(figname)))
        plt.close()

    def save_one_residual(self, y, figname):
        y =  self.normalize_array(y)
        y = np.squeeze(y)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(y, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.set_position([0, 0, 1, 1])  # 设置图像的边界为满画布，去除白边
        f.savefig(os.path.join(self.save_path, 'fig', '{}_residual.png'.format(figname)))
        plt.close()
        
    # def get_iteration_dataset(self,projdata,reference,reference2,res,res2,batch_index):
    #     #全部归一化
    #     projdata = self.normalize_array(projdata)
    #     reference = self.normalize_array(reference)
    #     reference2 = self.normalize_array(reference2)
    #     #res = self.normalize_array(res)
    #     #res2 = self.normalize_array(res2)
    #     #变成二维numpy格式
    #     projdata = np.squeeze(projdata)
    #     reference = np.squeeze(reference)
    #     reference2 = np.squeeze(reference2)
    #     res = np.squeeze(res)
    #     res2 = np.squeeze(res2)
    #     #确定阶段二训练数据集保存路径
    #     iter_mat_path = '/home/lqg/D/24/wyy/wyy/IRON_V1.0/rolling(IRON)/stage2_Iteration/test_full/{}.mat'.format(batch_index)
    #     sio.savemat(iter_mat_path,{'proj':projdata})
    #     print(f"appended proj to {iter_mat_path}")
    #     #读取现有mat文件
    #     mat_contents = sio.loadmat(iter_mat_path)
    #     #再插入
    #     mat_contents['reference'] = reference
    #     print(f"appended reference to {iter_mat_path}")
    #     mat_contents['reference2'] = reference2
    #     print(f"appended reference2 to {iter_mat_path}")
    #     mat_contents['data'] = res
    #     print(f"appended data to {iter_mat_path}")
    #     mat_contents['data2'] = res2
    #     print(f"appended data2 to {iter_mat_path}")
    #     sio.savemat(iter_mat_path, mat_contents ,do_compression=False)
    #     print('五个数据已保存')

    def save_IMA(self, pred, figname):
       import pydicom
       from pydicom.dataset import FileDataset
       from pydicom.uid import ImplicitVRLittleEndian

       template_path = "D:/24\wyy\wyy\wyy/3Dlimitangle/10L\L067/full_1mm\L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA"
       ds = pydicom.dcmread(template_path)

       # 获取模板的原始元数据
       original_position = ds.ImagePositionPatient  # 原始X,Y,Z位置（例如：[0.0, 0.0, 0.0]）
       slice_thickness = float(ds.SliceThickness)    # 层厚（如1mm）
       #spacing_between_slices = float(ds.SpacingBetweenSlices)  # 层间距（通常等于层厚）

       # 动态计算当前切片的Z轴位置
       # 假设figname是切片索引（从0开始），则Z = original_position[2] + figname * spacing_between_slices
       z_position = original_position[2] + figname * slice_thickness
       new_position = [original_position[0], original_position[1], z_position]
       ds.ImagePositionPatient = new_position  # 更新图像位置
       ds.SliceLocation = z_position           # 更新切片Z位置

       # 其他元数据更新
       ds.InstanceNumber = int(figname)        # 切片序号（必须唯一）
       ds.SOPInstanceUID = pydicom.uid.generate_uid()  # 唯一实例UID

       # 处理像素数据（保持原有逻辑）
       if pred.dtype != np.uint16:
           pred = (pred / pred.max() * 65535).astype(np.uint16)
       ds.PixelData = pred.tobytes()

       # 保存文件
       output_dir = os.path.join(self.save_path, 'IMA')
       os.makedirs(output_dir, exist_ok=True)
       output_path = os.path.join(output_dir, f"{figname}.IMA")
       ds.save_as(output_path, write_like_original=False)  # 确保元数据正确写入

       # 验证元数据（可选）
       print(f"Saved {output_path} with ImagePositionPatient={ds.ImagePositionPatient}, SliceLocation={ds.SliceLocation}")

    def test(self):
        class Logger(object):
            #作用：这个类用于将输出同时写入控制台和文件。
            #__init__ 方法：初始化时打开一个日志文件，并将标准输出重定向到这个文件。
            #write 方法：将消息写入控制台和日志文件。
            #flush 方法：确保缓冲区中的数据被写入文件
            def __init__(self, filename="Default.log"):
                self.terminal = sys.stdout
                self.log = open(filename,"a")
 
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
 
            def flush(self):
                pass
        path = os.path.abspath(os.path.dirname(__file__))
        type = sys.getfilesystemencoding()
        sys.stdout = Logger('test_result.txt')
        print(path)
        #print(os.path.dirname(__file__))
        print('------------------')

        
        loss_shallow=0 #累计测试集上浅层的总损失\
        loss_deep=0#累计测试集上的深层总损失
        losstest=0 #累计测试集上的总损失
        ern1=0 #累计 ERN（Error Normalized
        psnr1=0 #累计 PSNR（Peak Signal-to-Noise Ratio
        ssim1=0 #累计 SSIM（Structural Similarity Index）
        time_cost=0#时间消耗初始化
        for batch_index, data in enumerate(self.test_data):
            input_data, label_data,input_data2,label_data2,projdata= data
            if cuda:
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                label_data2 = label_data2.cuda()
                input_data2 = input_data2.cuda()
                projdata =projdata.cuda()

            with torch.no_grad():
                strat_time = time.time()
                output1,output2= self.model(input_data)
                end_time = time.time()
                time_cost += end_time - strat_time

            #计算损失
            res1 = output1.cpu().numpy()
            res2= output2.cpu().numpy()
            reference = label_data.cpu().numpy()
            reference2 = label_data2.cpu().numpy()
            inputs = input_data.cpu().numpy()
            input1 = input_data.cpu().numpy()
            projdata = projdata.cpu().numpy()
            #self.save_IMA(res2,batch_index)
            # res = (res1 + res2)/2
            # output = (output1 + output2)/2

            #为阶段二做数据集的准备
            #self.get_iteration_dataset(projdata,reference,reference2,res,res2,batch_index)
            

            #深层损失
            #losstest=losstest+torch.sum((label_data-output2)*(label_data-output2)/10/64/64)

            #为阶段二做数据集的准备
            #self.get_iteration_dataset(projdata,reference,reference2,res,res2,batch_index)
            output = (self.displaywin(output2) / 255).view(-1,256,256).cpu().numpy()#浅层输出结果
            label = (self.displaywin(label_data) / 255).view(-1,256,256).cpu().numpy()
            #新添加
            reference = (self.displaywin(label_data)/255).view(-1,256,256).cpu().numpy()
            inputs = (self.displaywin(input_data) / 255).view(-1,256,256).cpu().numpy()
            # output = (self.displaywin(output2) / 255).view(-1,64,64).cpu().numpy()#浅层输出结果
            # label = (self.displaywin(label_data) / 255).view(-1,64,64).cpu().numpy()
            # #新添加
            # reference = (self.displaywin(label_data)/255).view(-1,64,64).cpu().numpy()
            # inputs = (self.displaywin(input_data) / 255).view(-1,64,64).cpu().numpy()
            
            psnr = np.zeros(output.shape[0])
            ssim = np.zeros(output.shape[0])
            psnr_origial = np.zeros(output.shape[0])
            ssim_origial = np.zeros(output.shape[0])
            #单张图片指标
            psnr_onlyone_pred = 0
            ssim_onlyone_pred = 0
            psnr_onlyone_original = 0
            ssim_onlyone_original = 0
            for i in range(output.shape[0]):
                count=(batch_index)*output.shape[0]+i
                #计算模型预测指标
                psnr[i] = metrics.peak_signal_noise_ratio(label[i], output[i], data_range=1.0)
                ssim[i] = metrics.structural_similarity(label[i], output[i], data_range=1.0)
                #计算原始未经过修正有限角的指标
                psnr_origial[i] = metrics.peak_signal_noise_ratio(label[i], inputs[i], data_range=1.0) 
                ssim_origial[i] = metrics.structural_similarity(label[i], inputs[i], data_range=1.0)               
                #计算 ERN：计算每张图像的 ERN（Error Normalized
                norm21 = np.linalg.norm(output[i]-label[i])
                norm22 = np.linalg.norm(reference[i])
                ern=norm21/norm22
                #print(ern)
                # print("count:%f, psnr: %f, ssim: %f, ern: %f" % (count, psnr[i], ssim[i],ern))
                #累加 ERN、PSNR 和 SSIM 的值
                psnr1=psnr1+psnr[i]
                ssim1=ssim1+ssim[i]
                
                ern1=ern1+ern

                psnr_onlyone_pred = psnr_onlyone_pred + psnr[i]
                ssim_onlyone_pred = ssim_onlyone_pred + ssim[i]
                psnr_onlyone_original = psnr_onlyone_original + psnr_origial[i]
                ssim_onlyone_original = ssim_onlyone_original + ssim_origial[i]
                pred_result = (psnr_onlyone_pred/4 , ssim_onlyone_pred/4)
                original_result = (psnr_onlyone_original/4 , ssim_onlyone_original/4)

            self.save_fig(input1, reference2, res2, batch_index , original_result , pred_result)
            self.save_one_fig(res2, batch_index)
            self.save_one_residual(reference2-res2, batch_index)
                
                # sio.savemat(res_name[i], {'data':res[i,0], 'psnr':psnr[i], 'ssim':ssim[i],'psnrfbp':psnrfbp[i], 'ssimfbp':ssimfbp[i],'reference':reference[i,0],'inputs':inputs[i,0],'fullfbp':fullfbp1[i,0]})
                #注释掉了指标存储mat文件
                #sio.savemat(res_name[i], {'data':res[i,0]  ,'data2':res2[i,0]  ,'psnr':psnr[i], 'ssim':ssim[i],'reference':reference[i,0],'reference2':reference2[i,0],'proj':projdata[i,0]})
                #删除了'proj':projdata[i,0]

        #计算平均值        
        # psnr2=psnr1/400
        # ssim2=ssim1/400
        # ern2=ern1/400#391张测试改成测100张
        psnr2=psnr1/24
        ssim2=ssim1/24
        ern2=ern1/24#391张测试改成测100张
        # psnr2=psnr1/140
        # ssim2=ssim1/140
        # ern2=ern1/140#391张测试改成测100张
        print(" [loss_shallow: %f, loss_deep:%f, loss_test:%f,avgpsnr:%f,avgssim:%f,avgern:%f]" % ( loss_shallow, loss_deep, losstest,psnr2,ssim2,ern2))
        print("total_time: {}".format(time_cost))    

        #绘制热力图
        for batch_index, data in enumerate(self.test_data):
            input_data, label_data,input_data2,label_data2,projdata= data
            if cuda:
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                label_data2 = label_data2.cuda()
                input_data2 = input_data2.cuda()
                projdata =projdata.cuda()

            # 计算热力图（启用梯度计算）
            self.model.eval()  # 确保模型处于评估模式
            #input_for_gradcam = input_data[0].unsqueeze(0).detach().requires_grad_(True)
            heatmap = visualize_attention(self.model, input_data[0], 'module.block1.bottleneck_4.wtconv')
            # 保存热力图为PNG文件
            # 将原图转换为可视化格式
            original_img = input_data[0].cpu().numpy()
            original_img = self.displaywin(original_img) / 255.0
            original_img = np.squeeze(original_img)
        
            # 确保图像范围在0-1之间
            original_img = np.clip(original_img, 0, 1)
        
            # 调整热力图大小以匹配原图
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
            # 融合热力图和原图 (透明度0.4)
            overlay = cv2.addWeighted(
                (original_img * 255).astype(np.uint8), 
                0.4,
                heatmap_resized, 
                0.6, 
                0
            )
        
            # 保存融合图像
            overlay_path = os.path.join(self.save_path, 'fig', f'overlay_{batch_index}.png')
            cv2.imwrite(overlay_path, overlay)
            print(f"Saved overlay image to {overlay_path}")
        
            # 也可以保存单独的热力图
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            heatmap_path = os.path.join(self.save_path, 'fig', f'heatmap_{batch_index}.png')
            cv2.imwrite(heatmap_path, heatmap_bgr)
            print(f"Saved heatmap to {heatmap_path}")
            

if __name__ == "__main__":
    
    network = net()   
    network.train()
    network.test()