from train import Trainer
from test import Tester 
from model import LeNet
from dataset import DataSet, transform, train_set, train_loader, test_set, test_loader

import torch as t 
import torch.nn as nn 
import torchvision as tv 
from torch import optim
from torch.autograd import Variable
import argparse


def main(args):

    # CIFAR-10的全部类别，一共10类
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 数据集
    dataSet = DataSet(transform, train_set, train_loader, test_set, test_loader, classes)
    # 网络结构
    net = LeNet()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # SGD优化器
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)

    
    # 模型的参数保存路径
    model_path = "./model/state_dict"

    if args.do_train:
        trainer = Trainer(net, criterion, optimizer, dataSet.train_loader, args)
        trainer.train(epochs=args.epoch)
        t.save(net.state_dict(), model_path)
    
    if args.do_eval:
        model = LeNet()
        model.load_state_dict(t.load(model_path))
        model.eval()
        tester = Tester(dataSet.test_loader, model)
        tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 超参数
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epochs.")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="The momentum of the SGD optimizer.")

    # 输出日志 / 保存模型的步长
    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")

    # bool值默认为false，当命令中包含如下参数时则bool值变为true
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--is_shuffle", action="store_false", help="Whether shuffle the data samples or not.")
    
    args = parser.parse_args()
    main(args)