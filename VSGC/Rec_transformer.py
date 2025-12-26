## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from CAFM_Top import CATK



##########################################################################
## Layer Norm
#这两个函数 to_3d 和 to_4d 是用于在 PyTorch 张量之间进行维度变换的辅助函数。
# 它们使用了 rearrange 函数，这是来自 einops 库的一个强大工具，用于灵活地重排张量的维度。
#b是batch size , c是通道数，h是高度，w是宽度
def to_3d(x):
    #输入4d张量，将其重排为 3d 张量，其中 h 和 w 为输入张量的高度和宽度。
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    ##输入3d张量，将其重排为 4d 张量，其中 h 和 w 为输入张量的高度和宽度。
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

#无偏置的层归一化（Layer Normalization）模块，继承自 nn.Module。这个类的主要作用是对输入张量进行归一化处理，
# 并通过一个可学习的权重参数进行缩放
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        #normalized_shape：表示需要归一化的维度形状
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            #如果 normalized_shape 是一个整数，则将其转换为一个单元素的元组。
            normalized_shape = (normalized_shape,)
        #将 normalized_shape 转换为 torch.Size 对象
        normalized_shape = torch.Size(normalized_shape)

        #确保 normalized_shape 的长度为 1，即只对最后一个维度进行归一化
        assert len(normalized_shape) == 1

        #一个可学习的权重参数，初始化为全 1 的张量，形状与 normalized_shape 相同
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        #计算输入张量 x 在最后一个维度上的方差，keepdim=True 表示保留方差计算后的维度，unbiased=False 表示使用有偏估计
        sigma = x.var(-1, keepdim=True, unbiased=False)
        #x / torch.sqrt(sigma + 1e-5)：对输入张量 x 进行归一化处理，1e-5 是一个小的常数，用于防止除零错误
        #* self.weight：将归一化后的张量乘以可学习的权重参数 self.weight，实现缩放操作
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    #dim：表示需要归一化的维度大小
    #LayerNorm_type：表示使用的层归一化类型，可以是 'BiasFree' 或其他值（默认为带有偏置的层归一化）
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        '''
        to_3d(x)：将输入张量 x 从 4D 张量（形状为 [b, c, h, w]）转换为 3D 张量（形状为 [b, h*w, c]）。
        self.body(to_3d(x))：将 3D 张量传递给选择的层归一化模块进行归一化处理。
        to_4d(..., h, w)：将归一化后的 3D 张量重新转换为 4D 张量（形状为 [b, c, h, w]），恢复原始的高度和宽度
        '''
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
#主要功能是通过一系列卷积操作和非线性变换，对输入特征进行处理和变换
class FeedForward(nn.Module):
    #dim：表示输入特征的维度大小。
    #ffn_expansion_factor：表示隐藏层特征维度相对于输入特征维度的扩展因子。
    #bias：表示是否在卷积层中使用偏置项
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        #hidden_features = int(dim * ffn_expansion_factor)：根据输入特征维度和扩展因子计算隐藏层特征维度
        hidden_features = int(dim*ffn_expansion_factor)
        #self.project_in：一个1x1卷积层，将输入特征维度从 dim 扩展到 hidden_features * 2
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        #一个深度可分离卷积层（Depthwise Separable Convolution），输入和输出特征维度均为 hidden_features * 2。
        # 深度可分离卷积通过将标准卷积分解为深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）来减少计算量
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        #self.project_out：一个1x1卷积层，将特征维度从 hidden_features 缩减回 dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
#这种结构在Transformer模型中用于捕捉输入特征之间的依赖关系，增强模型的表达能力
#多头自注意力机制（Multi-Head Self-Attention Mechanism）
class Attention(nn.Module):
    #bias: 是否在卷积层中使用偏置项
    #dim: 输入特征的维度大小
    #num_heads: 注意力头的数量，决定了并行处理的不同子空间数量
    #temperature: 一个可学习的温度参数，形状为 (num_heads, 1, 1)，用于调整注意力分数的尺度
    #qkv: 一个 1x1 卷积层，将输入特征映射为查询（Q）、键（K）和值（V）三个部分，每个部分的维度都是 dim
    #qkv_dwconv: 一个深度可分离卷积层，对 Q、K、V 进行进一步的特征提取
    #project_out: 一个 1x1 卷积层，将多头注意力机制的输出映射回原始特征维度 dim。
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        #x: 输入的特征图，形状为 (b, c, h, w)，其中 b 是批量大小，c 是通道数，h 和 w 分别是高度和宽
        b,c,h,w = x.shape

        #使用 qkv 卷积层将输入特征映射为 Q、K、V 三个部分，形状为 (b, 3*dim, h, w)
        #使用 qkv_dwconv 深度可分离卷积层对 Q、K、V 进行进一步的特征提取，保持形状不变
        qkv = self.qkv_dwconv(self.qkv(x))
        #将 Q、K、V 沿通道维度拆分为三个独立的张量，每个张量的形状为 (b, dim, h, w)。
        q,k,v = qkv.chunk(3, dim=1)  

        #使用 rearrange 函数将 Q、K、V 的形状从 (b, dim, h, w) 变换为 (b, head, c, h*w)，
        # 其中 c = dim // num_heads，即每个注意力头处理的特征维度。
        #这样做的目的是为了并行处理多个注意力头 ，提高计算效率。
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        #对 Q 和 K 进行 L2 归一化，使得它们的范数为 1，有助于稳定训练过程
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        #计算 Q 和 K 的点积，得到注意力分数矩阵 attn，形状为 (b, head, h*w, h*w)。
        #使用 temperature 参数调整注意力分数的尺度。
        #对注意力分数矩阵进行 softmax 操作，得到注意力权重矩阵 attn，形状不变。
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        #使用注意力权重矩阵 attn 对 V 进行加权求和，得到注意力机制的输出 out，形状为 (b, head, c, h*w)。
        out = (attn @ v)

        #使用 rearrange 函数将注意力机制的输出 out 的形状从 (b, head, c, h*w) 变换回 (b, dim, h, w)。
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        #使用 project_out 卷积层将注意力机制的输出映射回原始特征维度 dim，得到最终的输出 out，形状为 (b, dim, h, w)
        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CATK(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
# -*- coding: utf-8 -*-

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)

class Pixel_UnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(Pixel_UnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body =nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.body1= Pixel_UnShuffle(2)
        
    def forward(self, x):
        return self.body1(self.body(x))

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class EdgeLayer(nn.Module):
    def __init__(self):
        super(EdgeLayer, self).__init__()
        # Define vertical and horizontal Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_m = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        sobel_n = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
    
        # Expand dimensions to match the expected input shape for convolution
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_m = sobel_m.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_n = sobel_n.view(1, 1, 3, 3).repeat(1, 1, 1, 1)   
        
        # Register the Sobel filters as buffers
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_m', sobel_m)
        self.register_buffer('sobel_n', sobel_n)
        

    def forward(self, x):
        # Apply the Sobel filters
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        edges_m = F.conv2d(x, self.sobel_m, padding=1)
        edges_n = F.conv2d(x, self.sobel_n, padding=1)
        
        # Concatenate the edge maps along the channel dimension
        edges = x + edges_x + edges_y + edges_m + edges_n
        return edges

class EnhancedEdgeLayer(nn.Module):
    def __init__(self, in_channels=1, use_scharr=True):
        super(EnhancedEdgeLayer, self).__init__()
        self.in_channels = in_channels
        
        # 基础Sobel滤波器
        self.register_buffer('sobel_x', self._create_kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        self.register_buffer('sobel_y', self._create_kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        
        # Scharr滤波器（对方向更敏感）
        if use_scharr:
            self.register_buffer('scharr_x', self._create_kernel([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]))
            self.register_buffer('scharr_y', self._create_kernel([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]))
        
        # 对角方向滤波器
        self.register_buffer('diag1', self._create_kernel([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]))  # 45°
        self.register_buffer('diag2', self._create_kernel([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]))  # 135°
        
        # 多尺度梯度检测（3x3和5x5）
        self.laplacian = self._create_kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.gradient5x5 = self._create_kernel([
            [-1, -2, 0, 2, 1],
            [-2, -3, 0, 3, 2],
            [-3, -4, 0, 4, 3],
            [-2, -3, 0, 3, 2],
            [-1, -2, 0, 2, 1]
        ])
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(9 if use_scharr else 7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 残差连接
        self.residual = nn.Conv2d(in_channels, 16, kernel_size=1) if in_channels != 16 else nn.Identity()

    def _create_kernel(self, kernel):
        k = torch.tensor(kernel, dtype=torch.float32)
        k = k.view(1, 1, *k.shape).repeat(self.in_channels, 1, 1, 1)
        return k

    def apply_kernels(self, x, kernels):
        return [F.conv2d(x, k, padding=k.shape[-1]//2, groups=self.in_channels) for k in kernels]

    def forward(self, x):
        # 多方向梯度检测
        gradients = self.apply_kernels(x, [
            self.sobel_x, self.sobel_y,
            self.diag1, self.diag2,
            self.laplacian, self.gradient5x5
        ])
        
        # 添加Scharr梯度（如果启用）
        if hasattr(self, 'scharr_x'):
            scharr_grads = self.apply_kernels(x, [self.scharr_x, self.scharr_y])
            gradients.extend(scharr_grads)
        
        # 添加原始输入作为参考
        gradients.append(x)
        
        # 拼接所有特征图
        edge_features = torch.cat(gradients, dim=1)
        
        # 特征融合
        fused = self.fusion_conv(edge_features)
        
        # 残差连接
        residual = self.residual(x)
        return fused + residual

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 48,
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = True,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()
        self.layer =  EdgeLayer()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        #添加卷积层规范通道数
        #self.conv1 = nn.Conv2d(inp_channels, 32, 3, 1, 1)
        #self.conv2 = nn.Conv2d(32, 1, 3, 1, 1)


    def forward(self, inp_img):
        inp_img = self.layer(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        #out_dec_level1 = self.conv1(out_dec_level1)
        #out_dec_level1 = self.conv2(out_dec_level1)
        return out_dec_level1


