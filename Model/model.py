import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class conv_block(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
#                nn.BatchNorm2d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                #default: mode='fan_in', nonlinearity='leaky_relu'
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)

        return x

# 配准网络 U-Net
class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf):
        super(U_Network, self).__init__()

        self.dim = dim
        self.enc_nf = enc_nf    # [16, 32, 32, 32]   encoder


        # Encoder     enc_nf = [2,16,32,32,64,64]
        self.input_conv = conv_block(enc_nf[0],enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block(enc_nf[-1],           dec_nf[0])
        self.up2 = conv_block(dec_nf[0]+enc_nf[4],  dec_nf[1])
        self.up3 = conv_block(dec_nf[1]+enc_nf[3],  dec_nf[2])
        self.up4 = conv_block(dec_nf[2]+enc_nf[2],  dec_nf[3])

        self.same_conv = conv_block(dec_nf[3]+enc_nf[1], dec_nf[4])
        self.out_conv = nn.Conv2d(
                dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        # 上采样    
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    # 网络的输入格式是（batch_size,channels,rows,cols)
    # [B, C, D, W, H]           而2D图像为(B,C,W,H)
    def forward(self, src, tgt):

        print("cat前图像:",src.shape)
        x = torch.cat([src, tgt], dim=1)    # 输入的是fixed图像与moving图像的拼接,通道加倍   (192, 160)
        print("cat后图像:",x.shape)

        # 下采样    (左半边)
        skip1 = self.input_conv(x)  
        print("skip1",skip1.shape)   
        skip2 = self.down1(skip1)
        print("skip2",skip2.shape)
        skip3 = self.down2(skip2)
        print("skip3",skip3.shape)
        skip4 = self.down3(skip3)
        print("skip4",skip4.shape)
        x = self.down4(skip4)

        # 上采样    (右半边)
        x = self.up1(x)
        print("up1",x.shape)
        x = self.upsample(x)
        print("up1_sample",x.shape)
        x = torch.cat((x, skip4), dim=1)

        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat((x, skip3), dim=1)

        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat((x, skip2), dim=1)

        x = self.up4(x)
        x = self.upsample(x)
        x = torch.cat((x, skip1), dim=1)
        print("up4",x.shape)

        x = self.same_conv(x)
        print("same_conv",x.shape)
        flow = self.out_conv(x)
        print("out_conv",x.shape)

        return flow



# 空间变换网络STN
class SpatialTransformer(nn.Module):        #只需要包含网格生成器 与 采样器
    def __init__(self, img_size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.img_size = img_size
        H, W = img_size, img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]
        
        # # Create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)  # y, x, z
        # grid = torch.unsqueeze(grid, 0)  # add batch
        # grid = grid.type(torch.FloatTensor)
        # self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        print("Now is STM网络's show time!!!!")
        print("src:",src.shape,"  flow:",flow.shape)
        grid = self.grid.repeat(flow.shape[0],1,1,1)#[bs, 2, H, W]
        if img.is_cuda:
            grid = grid.cuda()

        vgrid = Variable(grid, requires_grad = False) + flow
 
        vgrid = 2.0*vgrid/(self.img_size-1)-1.0 #max(W-1,1)
 
        vgrid = vgrid.permute(0,2,3,1) 
        #output = F.grid_sample(src, vgrid)
        # new_locs = self.grid + flow     
        # # shape = flow.shape[2:]                                # 我日你妈的!!!!
        # # # Need to normalize grid values to [-1, 1] for resampler
        # # for i in range(len(shape)):
        # #     new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # # if len(shape) == 2:
        # #     new_locs = new_locs.permute(0, 2, 3, 1)
        # #     new_locs = new_locs[..., [1, 0]]
        # # elif len(shape) == 3:
        # #     new_locs = new_locs.permute(0, 2, 3, 4, 1)
        # #     new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, vgrid, mode=self.mode) # 对moving图像进行sample,得到目标输出图像(即配准后的图像)

