from train import Trainer
from test import Tester 
from model import LeNet
from dataset import DataSet, transform, train_set, train_loader, test_set, test_loader

import torch as t 
import torch.nn as nn 
import torchvision as tv 
from torch import optim
from torch.autograd import Variable

# CIFAR-10的全部类别，一共10类
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 数据集
dataSet = DataSet(transform, train_set, train_loader, test_set, test_loader, classes)
# 网络结构
net = LeNet()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# SGD优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainer = Trainer(net, criterion, optimizer, dataSet.train_loader)
tester = Tester(dataSet.test_loader, net)

trainer.train(epochs=10)
tester.test()

