import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import Rec_transformer
from Rec_transformer import Restormer
from wtconv.wtconv2d import WTConv2d
#--------------------------------------------------------------------------------------------------------
class EdgeLayer(nn.Module):
    def __init__(self):
        super(EdgeLayer, self).__init__()
        # Define vertical and horizontal Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Expand dimensions to match the expected input shape for convolution
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        
        # Register the Sobel filters as buffers
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # Apply the Sobel filters
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Concatenate the edge maps along the channel dimension
        edges = x + edges_x + edges_y
        return edges

################################################
#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

#################################################
#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], dim=1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output
    
class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class conv_block1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block1,self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, 128, kernel_size=3, stride=1, padding=1,bias=True),
            #WTConv2d(128, 128, kernel_size=5,stride=1),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1,stride=1,padding=0,bias=True),
            #WTConv2d(128, 128, kernel_size=5,stride=1),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1,stride=1,padding=0,bias=True),
            #WTConv2d(128, 128, kernel_size=5,stride=1),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #WTConv2d(128, 128, kernel_size=5, stride=1),
            nn.Conv2d(128, ch_out, kernel_size=3, stride=1, padding=1,bias=True),
            
        )
        # 添加CBAM (注意通道数匹配最后输出32)
        #self.cbam = CBAMBlock(32)  # 新增

    def forward(self,x):
        x = self.conv(x)
        #x = self.cbam(x)
        return x



class dNN(nn.Module):

    def __init__(self):
        super(dNN, self).__init__()

        self.block1=conv_block1(32,32)
        self.block2=conv_block1(64,32)
        self.block3=conv_block1(96,32)
        self.block4=conv_block1(128,32)
        self.block5=conv_block1(160,32)
        self.block6=conv_block1(192,32)
        self.block7=conv_block1(224,32)
        self.block8=conv_block1(256,32)

    def forward(self, x):
        
        x1=self.block1(x)
        x2= torch.cat((x1, x), dim=1)
        x3=self.block2(x2)
        x4= torch.cat((x3, x2), dim=1)
        x5=self.block3(x4)
        x6= torch.cat((x5, x4), dim=1)
        x7=self.block4(x6)
        x8=torch.cat((x7, x6), dim=1)
        
        x9=self.block5(x8)
        x10=torch.cat((x9, x8), dim=1)
        x11=self.block6(x10)
        x12=torch.cat((x11, x10), dim=1)
        x13=self.block7(x12)
        x14=torch.cat((x13, x12), dim=1)
        x15=self.block8(x14)
        x16=torch.cat((x15, x14), dim=1)

        return x16

#------------------------------------------------------------------------------------------------------------

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=True)
            
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class deconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(deconv_block,self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_in, kernel_size=5,stride=1,padding=2,output_padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
            nn.ConvTranspose2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,output_padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
            nn.ConvTranspose2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,output_padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_in),
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,output_padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_out)  
        )


    def forward(self,x):
        x = self.deconv(x)
        return x



class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super(BottleneckBlock, self).__init__()
        self.wtconv = WTConv2d(channels, channels, kernel_size=5, wt_levels=1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.wtconv(x)
        out = self.conv(out)
        return self.relu(out + x)  # 残差连接



class DDNET(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(DDNET,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #增加边缘检测层
        self.edge_layer = EdgeLayer()
        # 在边缘检测后添加CBAM
        #self.edge_cbam = CBAMBlock(32)  # 输入通道为1

        #增加CBAM模块，在跳跃连接处添加CBAM
        #self.cbam_skip1 = CBAMBlock(64)  # cat后的通道数
        #self.cbam_skip2 = CBAMBlock(64)
        #self.cbam_skip3 = CBAMBlock(64)

        self.dnn=dNN()
        self.conv=conv_block(288,32)
        self.conv_to_WT = conv_block(32,4)
        self.WT_to_conv = conv_block(4,32)
        self.bottleneck = BottleneckBlock(4)
        self.bottleneck_4 = BottleneckBlock(32)

        self.Up5 = up_conv(ch_in=32,ch_out=32)
        self.Up_conv5 = deconv_block(ch_in=64, ch_out=32)

        self.Up4 = up_conv(ch_in=32,ch_out=32)
        self.Up_conv4 = deconv_block(ch_in=64, ch_out=32)
        
        self.Up3 = up_conv(ch_in=32,ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        
        self.Up2 = up_conv(ch_in=32,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=1)
        self.conv11=nn.Conv2d(1, 32, kernel_size=7,stride=1,padding=3,bias=True)
        self.Conv77=WTConv2d(in_channels=1, out_channels=1, kernel_size=7,stride=1)

    def forward(self,x):
        # encoding path
        x_edge = self.edge_layer(x)
        #边缘检测层后加入CBAM
        x1 = self.Conv77(x_edge)
        x1 = self.conv11(x1)
        
        #x1 = self.edge_cbam(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.dnn(x2)
        x2= self.conv(x2)
        x2 = self.conv_to_WT(x2)
        x2 = self.bottleneck(x2)
        x2 = self.WT_to_conv(x2)

        x3 = self.Maxpool(x2)
        x3 = self.dnn(x3)
        x3 = self.conv(x3)
        x3 = self.conv_to_WT(x3)
        x3 = self.bottleneck(x3)
        x3 = self.WT_to_conv(x3)

        x4 = self.Maxpool(x3)
        x4 = self.dnn(x4)
        x4 = self.conv(x4)
        x4 = self.bottleneck_4(x4)
        x4 = self.bottleneck_4(x4)

        d4 = self.Up4(x4)#x4->d5,效果不行再改回去
        d4 = torch.cat((x3,d4),dim=1)
        #d4 = self.cbam_skip1(d4)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        #d3 = self.cbam_skip2(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        #d2 = self.cbam_skip3(d2)
        d2 = self.Up_conv2(d2)

        
        output=d2+x_edge

        return output

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class VSGC(nn.Module):
    def __init__(self):
        super(VSGC, self).__init__()
        #self.lambda1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.miu = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.lambda1.data.fill_(0.1)
        #self.miu.data.fill_(0.0)
        self.block1=DDNET()
        #加入了瓶颈层，减少DDNET和Restormer之间的互相影响
        #self.block_2=Bottleneck(inplanes=1, planes=32)
        self.block90=Restormer()
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1),nn.LeakyReLU(inplace=True))
        self.conv_last2 = nn.Conv2d(32, 1, 3, 1, 1)
        #self.relu = nn.ReLU(inplace=True)
        
    
    def forward(self, input_data):
        deep = self.block1(input_data)
        #shallow = self.block_2(shallow)
        deep = self.block90(deep)
        deep2 = self.conv_before_upsample(deep)
        deep2 = self.conv_last2(deep2)
        return deep , deep2

# class CAT(nn.Module):
#     def __init__(self, block_num):
#         super(CAT, self).__init__()
#         #views = kwargs['views']
#         #dets = kwargs['dets']
#         #width = kwargs['width']
#         #height = kwargs['height']
#         #dImg = kwargs['dImg']
#         #dDet = kwargs['dDet']
#         #Ang0 = kwargs['Ang0']
#         #dAng = kwargs['dAng']
#         #s2r = kwargs['s2r']
#         #d2r = kwargs['d2r']
#         #binshift = kwargs['binshift']
#         #scale = kwargs['scale']
#         #scan_type = kwargs['scan_type']
#         #options = torch.Tensor([views, dets, width, height, dImg, dDet, Ang0,dAng, s2r, d2r, binshift,scan_type])
        
#         self.block = nn.ModuleList([IterBlock() for i in range(int(block_num))])
#         #self.conv_before_upsample = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1),nn.LeakyReLU(inplace=True))
#         #self.upsample = Upsample(1, 32)#2->1
#         #self.conv_last2 = nn.Conv2d(32, 1, 3, 1, 1)
         
#     def forward(self, input_data):
#         for index, module in enumerate(self.block):
#             shallow, deep = module(input_data)
#         #x = self.conv_before_upsample(deep)
#         #x = self.conv_last2(self.upsample(x))   
#         return shallow, deep
#     # #两个参数的训练模型前项过程
#     # def forward(self, input_data, proj):
#     #     x = input_data
#     #     for index, module in enumerate(self.block):
#     #         x = module(x, proj)
#     #     x2 = self.conv_before_upsample(x)
#     #     x2 = self.conv_last2(self.upsample(x2))   
#     #     return x,x2