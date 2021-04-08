from struct import pack_into
import torch as t 
from model import LeNet 
from torch.autograd import Variable
from tqdm import tqdm, trange

class Trainer():
    def __init__(self, net, criterion, optimizer, train_loader, args):
        self.net = net 
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
    
    def train(self, epochs):
        for epoch in range(epochs):
            print("\n******** Epoch %d / %d ********\n" % (epoch + 1, epochs))
            running_loss = 0.0
            epoch_iterator = tqdm(self.train_loader, desc="Iteration", ncols=70)
            for i, data in enumerate(epoch_iterator):
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
                # if i % 2000 == 1999:
                #     print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0
            print('Epoch {} finish, loss: {}'.format(epoch + 1, running_loss / (i + 1)))
        print('\nFinish training\n')               
