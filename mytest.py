from torch.utils.data import DataLoader
from torchvision import transforms as T
from Model.datagenerators import Dataset
from torch.autograd import Variable
import torch


dataset = Dataset("./output")
img, label = dataset[0]
# for img, label in dataset:
#     print(img.size(), label)

device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
dataiter = iter(dataloader)

imgs,labels = next(dataiter)
print(imgs.size())
print(imgs[-1].size())


# input_moving = input_moving.to(device).float()
# input_moving = iter(DL).next()
#         print(input_moving.size())