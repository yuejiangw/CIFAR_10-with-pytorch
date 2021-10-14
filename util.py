# coding:utf-8
import sys
sys.path.append(".")
import os
import random
import logging
import torch as t
import numpy as np

def init_logger():
    """ 将日志信息输出到控制台
    Params:
        asctime: 打印日志的时间
        levelname: 打印日志级别
        name: 打印日志名字
        message: 打印日志信息 
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

def check_path(args):
    """ 检查目标目录是否存在，若不存在，则新建目录 """
    print("Start checking path...")
    if not os.path.exists(args.data_path):
        print("Creating data path...")
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        print("Creating model path...")
        os.makedirs(args.model_path)
    print("Check path done.")

def set_seed(args):
    """
    为了得到可重复的实验结果需要对所有随机数生成器设置一个固定的种子
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    if not args.no_cuda and t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)

def show_model(args):
    model_path = os.path.join(args.model_path, args.model_name)
    device = t.device("cuda:0" if t.cuda.is_available() and not args.no_cuda else "cpu")
    net = t.load(model_path, map_location=t.device(device))
    print(type(net))
    print(len(net))
    for k in net.keys():
        print(k)