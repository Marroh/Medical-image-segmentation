from U_Net import Data, UNet
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

dataset = Data(list_path='D:\Data_set\VOCdevkit\VOC2012\ImageSets\Segmentation/train.txt',
               data_root='D:\Data_set\VOCdevkit\VOC2012\JPEGImages',
                mask_root='D:\Data_set\VOCdevkit\VOC2012\SegmentationClass',
               transforms_=True)

data_loader = DataLoader(dataset=dataset, batch_size=5, pin_memory=True, shuffle=True, drop_last=True)

unet = UNet(3, 22)#两个参数分别是输入和输出两个矩阵的通道数。输入是RBG图有3个channel，输出有22个类别，所以大概是22个channel
# （这个数据集用1到20标注了20个类，用0标注背景，255标注勾画的边界每个像素的类别有22种可能）。

#unet.train()
#unet.load_state_dict(torch.load('C:/Users/Desktop/parameters.pt'))
#torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Have transformed the net to GPU')
    unet.to(device)

#print(list（unet.parameters())）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        data[0] = np.array(data[0]).transpose((0,3,1,2))
        inputs = torch.Tensor(data[0]).to(device)
        lables = data[1].type(torch.LongTensor).to(device)
        lables = torch.squeeze(lables)

        #print(inputs.size(), lables.size())
        inputs, lables = Variable(inputs), Variable(lables)

        optimizer.zero_grad()

        outputs = unet(inputs)

        #print(outputs)

        loss = criterion(outputs, lables)

        #print(inputs.size(), outputs.size(), lables.size())
        #size: [5, 3, x, y]    [5, 20, x, y]   [5, x, y]


        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print('[epoch:%d, num_of_batches:%3d] loss: %.3f' %(epoch+1, i+1, running_loss/49))
            running_loss = 0.0
print('Finish Training')

torch.save(unet.state_dict(), 'C:/Users/mamama9503/Desktop/parameters.pt')
print('Saved')
