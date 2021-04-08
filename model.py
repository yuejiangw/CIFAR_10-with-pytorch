import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class LeNet(nn.Module):
    """ 网络模型选用的是比较简单的LeNet模型 """
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1
        # 3表示输入图片的颜色通道为3（彩图），6为输出通道数，5表示卷积核大小
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1
        self.fc1 = nn.Linear(16*5*5, 120)
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class ResNet():
    pass