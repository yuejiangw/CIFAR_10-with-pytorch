from train import Trainer
from test import Tester 
from model import LeNet
from dataset import DataSet, DataBuilder 

import torch as t 
import torch.nn as nn 
import torchvision as tv 
from torch import optim
from torch.autograd import Variable
import argparse
import os


def main(args):

    # CIFAR-10的全部类别，一共10类
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 数据集
    data_builder = DataBuilder(args)
    dataSet = DataSet(data_builder.train_builder(), data_builder.test_builder(), classes)
    
    # 网络结构
    net = LeNet()
    
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # SGD优化器
    optimizer = optim.SGD(
        net.parameters(), 
        lr=args.learning_rate, 
        momentum=args.sgd_momentum, 
        weight_decay=args.weight_decay
    )

    
    # 模型的参数保存路径，默认为 "./model/state_dict"
    model_path = os.path.join(args.model_path, args.model_name)

    # 启动训练
    if args.do_train:
        trainer = Trainer(net, criterion, optimizer, dataSet.train_loader, args)
        trainer.train(epochs=args.epoch)
        t.save(net.state_dict(), model_path)
    
    # 启动测试
    if args.do_eval:
        if os.listdir(model_path) == []:
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        model = LeNet()
        model.load_state_dict(t.load(model_path))
        model.eval()
        tester = Tester(dataSet.test_loader, model)
        tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    # 数据集
    parser.add_argument("--num_workers", default=0, type=int, help="Thread number for training.")
    parser.add_argument("--is_download", default=True, type=bool, help="Download the datasets if there is no data.")

    # 路径
    parser.add_argument("--data_path", default="./data", type=str, help="The directory of the CIFAR-10 data.")
    parser.add_argument("--model_path", default="./model", type=str, help="The directory of the saved model.")
    parser.add_argument("--model_name", default="state_dict", type=str, help="The name of the saved model's parameters.")

    # 训练相关
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epochs.")
    
    # 超参数
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.001, type=int, help="Weight decay of SGD optimzer.")
    parser.add_argument("--sgd_momentum", default=0.8, type=float, help="The momentum of the SGD optimizer.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The Epsilon of Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    # 采取的动作
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    args = parser.parse_args()
    main(args)