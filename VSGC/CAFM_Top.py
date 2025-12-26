import torch
import torch.nn as nn
from einops import rearrange
# 论文题目：Learning A Sparse Transformer Network for Effective Image Deraining
# 中文题目：学习稀疏变换网络以有效进行图像去雨
# 论文链接：https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_a_Sparse_Transformer_Network_for_Effective_Image_Deraining_CVPR_2023_paper.pdf
# 官方github：https://github.com/cschenxiang/DRSformer
# 所属机构：南京理工大学计算机科学与工程学院，中国电子科技集团信息科学研究院
##  CATK model,基于上述论文改进实现
class CATK(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CATK, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.qkv_cafm = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv_cafm = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)
        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)
    def forward(self, x):
        b, c, h, w = x.shape
        # global SA
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        #local conv
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv_cafm(self.qkv_cafm(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)

        output =  out + out_conv
        return output