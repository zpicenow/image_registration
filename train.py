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
from torchvision import transforms
unloader = transforms.ToPILImage()
lcc = losses.LCC()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image



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
    # 加载之前训练的模型(指定轮数)
    pth_epoch = 230
    # 创建需要的文件夹并指定gpu
    make_dirs() 
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # 日志文件
    log_name = str(pth_epoch) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像
    input_fixed = Image.open("1.png").convert('L') #转灰度图
    input_fixed = transform(input_fixed) # 将图片(Image)转成Tensor，归一化至[0, 1]
    vol_size = input_fixed.shape   
    print(vol_size)
    input_fixed = torch.unsqueeze(input_fixed,0)
    input_fixed = input_fixed.to(device).float()
    print("fix图像:",input_fixed.shape)


    # 创建配准网络（UNet）和STN
    nf_enc = [2,16,32,32,64,64]
    nf_dec = [64,32,32,32,16,1]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    
    pre=torch.load(os.path.join('/home/mingjie/VoxelMorph-torch/Checkpoint',str(pth_epoch)+'.pth'))
    UNet.load_state_dict(pre)

    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()

    
    
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # 加载训练数据
    dataset = Dataset("./data/day7")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataiter = iter(dataloader)

    # Training loop.
    for i in range(pth_epoch, args.n_iter + 1):
        # 加载浮动图像并转换成Tensor        Generate the moving images and convert them to tensors.
        for idx, (input_moving, ref) in enumerate(dataloader):
            
            input_moving = input_moving.to(device).float()
            #print("浮动图像cat",input_moving.shape)
            y = input_fixed
            for ii in range(1,args.batch_size):
                y = torch.cat( [y,input_fixed], dim = 0)
            #print("固定图像cat",y.shape)    

            x = torch.cat( [input_moving,y], dim = 1)
            flow_m2f = UNet(x)
            m2f = STN(input_moving, flow_m2f)
  
            
            # Calculate loss
            #sim_loss = sim_loss_fn(m2f, input_fixed)    # 互信息损失
            sim_loss = lcc(m2f, input_fixed)    # 互信息损失
            grad_loss = grad_loss_fn(flow_m2f)          # 位移场平滑损失
            loss = sim_loss + args.alpha * grad_loss        
            print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)


            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % args.n_save_iter == 0  and (idx+1)*args.batch_size>=(dataset.__len__()-2):
                # Save model checkpoint
                save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
                torch.save(UNet.state_dict(), save_file_name)
                # Save images
                new_img = tensor_to_PIL(torch.squeeze(flow_m2f[0,:,:],dim=0))
                new_img.save("./Result/flow"+str(i)+".jpg")
                new_img = tensor_to_PIL(torch.squeeze(m2f[0,:,:],dim=0))
                new_img.save("./Result/registrated"+str(i)+".jpg")
                print("warped images have saved.")
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
