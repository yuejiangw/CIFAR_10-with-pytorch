import torch.nn as nn 
import torch.nn.functional as F 

class LeNet(nn.Module):
    """ Implements LeNet model. """

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

class Vgg16_Net(nn.Module):
    """ Implements vgg-16 model. """

    def __init__(self):
        super(Vgg16_Net, self).__init__()

        # 第一层: 2个卷积层和1个最大池化层
        self.layer1 = nn.Sequential(
            
            # (32-3+2)/1+1 = 32  32*32*64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (32-3+2)/1+1 = 32  32*32*64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (32-2)/2+1 = 16    16*16*64
            nn.MaxPool2d(2, 2)     
        )
        
        # 第二层: 2个卷积层和1个最大池化层
        self.layer2 = nn.Sequential(
            
            # (16-3+2)/1+1 = 16  16*16*128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),           
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (16-3+2)/1+1 = 16  16*16*128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # (16-2)/2+1 = 8    8*8*128
            nn.MaxPool2d(2, 2)                                                  
        )

        # 第三层: 3个卷积层和1个最大池化层
        self.layer3 = nn.Sequential(
            
            # (8-3+2)/1+1 = 8  8*8*256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (8-3+2)/1+1 = 8  8*8*256
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (8-3+2)/1+1 = 8  8*8*256
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # (8-2)/2+1 = 4    4*4*256
            nn.MaxPool2d(2, 2),                                                 
        )

        # 第四层: 3个卷积层和1个最大池化层
        self.layer4 = nn.Sequential(

            # (4-3+2)/1+1 = 4  4*4*512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (4-3+2)/1+1 = 4  4*4*512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (4-3+2)/1+1 = 4  4*4*512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # (4-2)/2+1 = 2    2*2*512
            nn.MaxPool2d(2, 2)                                                  
        )

        # 第五层: 3个卷积层和1个最大池化层
        self.layer5 = nn.Sequential(

            # (2-3+2)/1+1 = 2  2*2*512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (2-3+2)/1+1 = 2  2*2*512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (2-3+2)/1+1 = 2  2*2*512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),          
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # (2-2)/2+1 = 1    1*1*512
            nn.MaxPool2d(2, 2)                                                  
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(    
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
    
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x