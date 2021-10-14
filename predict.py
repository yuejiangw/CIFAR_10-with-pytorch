import torch as t 
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

class Predictor():
    def __init__(self, net, classes):
        self.net = net
        self.classes = classes
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.net.to(self.device)
    
    def predict_transform(self):
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),  # 转换成Tensor

            # 参考网上的参数进行的Normalization，效果比0.5更好
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform

    # 单张图片的预测
    def predict(self, img_path):
        transform = self.predict_transform()
        
        img = Image.open(img_path)
        img_tensor = transform(img).to(self.device)
        img_tensor = Variable(t.unsqueeze(img_tensor, 0).float(), requires_grad=False)

        with t.no_grad():
            self.net.eval()
            output = self.net(img_tensor)
            _, predicted = t.max(output, 1)
            print("下标: ", predicted)
            print("标签: ", self.classes[predicted])
            return predicted