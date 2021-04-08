import torch as t 
from torch.autograd import Variable
import torchvision as tv 
from dataset import test_loader

class Tester():
    def __init__(self, test_loader, net):
        self.test_loader = test_loader
        self.net = net

    def test(self):
        correct = 0 # 预测正确的图片数
        total = 0   # 总共的图片数
        for data in test_loader:
            images, labels = data
            outputs = self.net(Variable(images))
            _, predicted = t.max(outputs.data, 1)   # torch.max返回值为(values, indices)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

