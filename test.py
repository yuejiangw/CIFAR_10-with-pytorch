import torch as t 
from torch.autograd import Variable
import torchvision as tv 
from tqdm import tqdm

class Tester():
    def __init__(self, test_loader, net, args):
        self.test_loader = test_loader
        self.net = net
        self.device = t.device("cuda:0" if t.cuda.is_available() and not args.no_cuda else "cpu")
        self.net.to(self.device)

    def test(self):
        correct = 0 # 预测正确的图片数
        total = 0   # 总共的图片数
        self.net.eval() # 将net设置成eval模式
        for data in tqdm(self.test_loader, desc="Test Iteration", ncols=70):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net(Variable(images))
            _, predicted = t.max(outputs.data, 1)   # torch.max返回值为(values, indices)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
    
    def predict(self):
        pass