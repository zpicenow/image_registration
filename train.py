# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.model import U_Network, SpatialTransformer

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import transforms as T
from Model.datagenerators import Dataset

from PIL import Image






def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():    # 对应文件夹不存在的话就创建文件夹
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

transform = T.Compose([
                T.Resize(512), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                T.CenterCrop(512), # 从图片中间切出224*224的图片
                T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
                T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
            ])
def train():
    # 创建需要的文件夹并指定gpu
    make_dirs() 
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

     # 读入fixed图像
    input_fixed = Image.open("output/dog.png").convert('L') #转灰度图
    

    input_fixed = transform(input_fixed) # 将图片(Image)转成Tensor，归一化至[0, 1]

    vol_size = input_fixed.shape   # 三维图像尺寸  (160, 192, 160)
    print(vol_size)
    input_fixed = torch.unsqueeze(input_fixed,0)
    
    input_fixed = input_fixed.to(device).float()
    print("fix图像:",input_fixed.shape)
    # [B, C, D, W, H]
   



    # # 读入fixed图像
    # f_img = sitk.ReadImage(args.atlas_file)
    # input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]    #　(1, 1, 160, 192, 160)
    # print(input_fixed.shape)
    # vol_size = input_fixed.shape[2:]    # 三维图像尺寸  (160, 192, 160)
    # print(vol_size)
    # # [B, C, D, W, H]
    # input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)   #　(1, 1, 160, 192, 160)
    # print(input_fixed.shape)
    # input_fixed = torch.from_numpy(input_fixed).to(device).float()  #　torch.Size([1, 1, 160, 192, 160])



    # 创建配准网络（UNet）和STN
    nf_enc = [2,16,32,32,64,64]
    nf_dec = [64,32,32,32,16,1]

    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()


    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))
    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # 加载训练数据
    dataset = Dataset("./output")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    dataiter = iter(dataloader)


    # Training loop.
    for i in range(1, args.n_iter + 1):
        # 加载浮动图像并转换成Tensor        Generate the moving images and convert them to tensors.
        input_moving,_ = next(dataiter)
        print("浮动图像:",input_moving.size())
        
        # [B, C, D, W, H]
        input_moving = input_moving.to(device).float()

        # Run the data through the model to produce warp and flow field
        flow_m2f = UNet(input_moving, input_fixed)
        m2f = STN(input_moving, flow_m2f)

        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow_m2f)
        loss = sim_loss + args.alpha * grad_loss
        print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = str(i) + "_m.nii.gz"
            m2f_name = str(i) + "_m2f.nii.gz"
            save_image(input_moving, f_img, m_name)
            save_image(m2f, f_img, m2f_name)
            print("warped images have saved.")
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
