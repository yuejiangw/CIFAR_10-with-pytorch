import torch as t 
from model import LeNet 
from torch.autograd import Variable

class Trainer():
    def __init__(self, net, criterion, optimizer, train_loader):
        self.net = net 
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
    
    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # 输入数据
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                # 梯度清零
                self.optimizer.zero_grad()

                # forward + backward
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # 更新参数
                self.optimizer.step()

                # 打印训练信息
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finish training')               
