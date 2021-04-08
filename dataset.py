import torchvision as tv 
import torch as t 
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage


# # 可以将Tensor转换成PIL图像，可视化
# show = ToPILImage()

class DataSet():
    def __init__(self, train_loader, test_loader, classes):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = classes


class DataBuilder():
    def __init__(self, args) -> None:
        self.args = args
    
    # 数据预处理, 通过Compose将各个变换串联起来
    def transform_builder(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换成Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图片各个通道的像素归一化
        ])
        return transform
    
    def train_builder(self):
        # 训练集
        train_set = tv.datasets.CIFAR10(
            root='./data/',
            train=True,
            download=self.args.is_download,
            transform=self.transform_builder()
        )

        train_loader = t.utils.data.DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        return train_loader
    
    def test_builder(self):
        # 测试集
        test_set = tv.datasets.CIFAR10(
            root='./data/',
            train=False,
            download=self.args.is_download,
            transform=self.transform_builder()
        )

        # 由测试集创建对应的DataLoader
        test_loader = t.utils.data.DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        return test_loader