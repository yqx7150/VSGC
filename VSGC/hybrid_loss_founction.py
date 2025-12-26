import torch
import torch.nn as nn
from pytorch_msssim import ssim
import torchvision
from red_cnn import RED_CNN
import numpy as np
#import torch_radon24 as tr


def normalize_array(data):
    """
    对输入的 numpy 数组进行归一化处理，使其值范围在 0 到 1 之间

    :param data: 输入的 numpy 数组
    :return: 归一化后的数组
    """
    # 确保输入的是 numpy 数组
    if not isinstance(data, np.ndarray):
        raise ValueError("输入必须是 numpy 数组")
    # 归一化处理
    min_value = data.min()
    max_value = data.max()
    # 处理边界情况
    if max_value - min_value == 0:
        normalized_data = np.zeros_like(data)  # 如果所有值都相同，归一化后为全零
    else:
        normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data




class PerceptualLoss(nn.Module):
    def __init__(self, model='RED_CNN', layers=[0,1,2,3,4], criterion=nn.L1Loss(), device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.model, self.layers = self.get_layers(model, layers)
        self.model.to(device)
        self.model.eval()

    def get_layers(self, model, layers):
        if model == 'vgg16':
            vgg = torchvision.models.vgg16(pretrained=True).features
            model = nn.Sequential()
            for i, layer in enumerate(list(vgg)):
                model.add_module(str(i), layer)
                if i in layers:
                    model.add_module(str(i) + "_act", nn.ReLU())
            return model, layers
        elif model == 'RED_CNN':
            #只要在这里添加自定义模型即可
            # custom_model = RED_CNN()
            # custom_model.load_state_dict(torch.load('D:/24\wyy\wyy\wyy\IRON_V1.0/rolling(IRON)\CAFM_Top-k_hybrid\REDCNN_29000iter.ckpt'))
            # # 请替换为实际的自定义模型路径
            custom_model = RED_CNN()
            state_dict = torch.load('/home/ly/wyy/Ours_WTConv_edgeloss_weight/red_cnn/xiaoshu_90.ckpt')
            # 移除 'module.' 前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            custom_model.load_state_dict(new_state_dict)

            # 冻结所有参数
            for param in custom_model.parameters():
                param.requires_grad = False

            model = nn.Sequential()
            for i, layer in enumerate(custom_model.children()):
                if(i<5):
                    model.add_module(str(i), layer)
                #if i in layers:
                    #model.add_module(str(i) + "_act", nn.ReLU())
            return model, layers
        else:
            raise ValueError(f"Unsupported model: {model}")

    def forward(self, x, y):
        with torch.no_grad():
            loss = 0
            for i, layer in enumerate(self.model):
                x = layer(x)
                y = layer(y)
                loss += self.criterion(x, y)
            return loss


# def PHYSICS_LOSS(pred, target):
#     mse = nn.MSELoss()
#     thetas = np.linspace(0, 2*np.pi, 720, endpoint=False)
#     radon_transform = tr.Radon(thetas=thetas, image_size=720, circle=False, det_count=None, filter="ramp", device="cuda")
#     pred = radon_transform(pred)
#     target = radon_transform(target)
#     physics_loss = mse(pred, target)
#     return physics_loss

class HybridLoss(nn.Module):
    def __init__(self, deep_weight=1.0, 
                 perceptual_weight=0.5,
                 ssim_weight=0.1,
                 model="RED_CNN",
                 corner_weight=2.0,
                 edge_weight=0.2):  # 新增边缘损失权重
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.deep_weight = deep_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.corner_weight = corner_weight
        self.edge_weight = edge_weight  # 边缘梯度损失权重
        self.perceptual_loss = PerceptualLoss(model=model, device='cuda')
        
        # 创建Sobel边缘检测算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 初始化Sobel滤波器权重
        sobel_kernel_x = torch.tensor([[-1, 0, 1], 
                                       [-2, 0, 2], 
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], 
                                       [0, 0, 0], 
                                       [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)
        
        # 将算子移动到GPU
        self.sobel_x = self.sobel_x.to('cuda')
        self.sobel_y = self.sobel_y.to('cuda')

    def create_corner_mask(self, img_size):
        """创建关注右上角和左下角的权重掩码"""
        mask = torch.ones(img_size, device='cuda')
        h, w = img_size[-2], img_size[-1]
        corner_size = (h // 4, w // 4)
        mask[..., :corner_size[0], -corner_size[1]:] *= self.corner_weight
        mask[..., -corner_size[0]:, :corner_size[1]] *= self.corner_weight
        return mask

    def compute_edge_loss(self, pred, target):
        """计算边缘梯度损失"""
        # 确保输入是单通道（灰度图）
        if pred.dim() == 4 and pred.size(1) > 1:
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)
        
        # 计算水平和垂直方向的梯度
        pred_grad_x = self.sobel_x(pred)
        pred_grad_y = self.sobel_y(pred)
        target_grad_x = self.sobel_x(target)
        target_grad_y = self.sobel_y(target)
        
        # 计算梯度幅度
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # 计算边缘损失
        edge_loss = self.mse_loss(pred_grad_mag, target_grad_mag)
        return edge_loss

    def forward(self, deep, target):
        # 创建角落权重掩码
        corner_mask = self.create_corner_mask(deep.shape)
        
        # 应用区域权重的深层损失
        weighted_mse = (deep - target) ** 2 * corner_mask
        deep_loss = weighted_mse.mean()
        
        # 计算感知损失
        perceptual_loss = self.perceptual_loss(deep, target)
        
        # 计算SSIM损失
        ssim_loss = 1 - ssim(deep, target, data_range=1.0)
        
        # 计算边缘梯度损失（新增）
        edge_loss = self.compute_edge_loss(deep, target)
        
        total_loss = (
            self.deep_weight * deep_loss +
            self.perceptual_weight * perceptual_loss + 
            self.ssim_weight * ssim_loss +
            self.edge_weight * edge_loss  # 添加边缘损失项
        )
        return total_loss, deep_loss, perceptual_loss, ssim_loss, edge_loss  # 返回新增的edge_loss

    
