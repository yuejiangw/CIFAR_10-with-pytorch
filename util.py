# coding:utf-8
import sys
sys.path.append(".")
import os
import random
import logging
import torch as t
import numpy as np

def init_logger():
    """将日志信息输出到控制台
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
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

def set_seed(args):
    """
    为了得到可重复的实验结果需要对所有随机数生成器设置一个固定的种子
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    if not args.no_cuda and t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)
