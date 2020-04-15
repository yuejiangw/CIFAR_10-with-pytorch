import torchvision as tv 
import torch as t 
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

class DataSet():
    def __init__(self, transform, train_set, train_loader, test_set, test_loader, classes):
        self.transform = transform
        self.train_set = train_set
        self.train_loader = train_loader
        self.test_set = test_set
        self.test_loader = test_loader
        self.classes = classes

# 可以将Tensor转换成PIL图像，可视化
show = ToPILImage()

# 数据预处理, 通过Compose将各个变换串联起来
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图片各个通道的像素归一化
])

# 训练集
train_set = tv.datasets.CIFAR10(
    root='./data/',
    train=True,
    download=True,
    transform=transform
)

# 由训练集创建对应的DataLoader
train_loader = t.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# 测试集
test_set = tv.datasets.CIFAR10(
    root='./data/',
    train=False,
    download=True,
    transform=transform
)

# 由测试集创建对应的DataLoader
test_loader = t.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=0
)