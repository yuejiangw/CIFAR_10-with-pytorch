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

class VGG_16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty list for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result



class ResNet():
    pass