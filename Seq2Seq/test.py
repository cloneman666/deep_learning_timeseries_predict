import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.utils import *  #每个函数内部的方法
from model.Seq2Seq_attention import *
from importlib import import_module  #动态加载不同的模块
import utils   #这个为计算时间的方法，为公共方法，所以定义在外面

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="Seq2Seq类模型进行时间序列预测")

    #选择模型即可
    parser.add_argument('--model_name',type=str,default='Seq2Seq_attention',help='choose a model Seq2Seq、Seq2Seq_attention、CNN_LSTM')

    args = parser.parse_args()

    return args


def main_Seq2Seq_Att():
    """
    该函数为训练Seq2Seq_Att模型的函数
    :return:
    """
    args = parse_args() #加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.'+ model_name)
    config = x.Config()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(config.dataroot, debug=False)

    # Initialize model
    print("==> Initialize Seq2Seq_attention model ...")
    model = x.Model(
        X,
        y,
        config
    )

    # Train
    print("==> Start training ...")
    model.train(model,config)  # model 输入进去用于保存模型

    # Prediction
    y_pred = model.test()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("./data/pic/1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("./data/pic/2.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("./data/pic/3.png")
    plt.close(fig3)
    print('Finished Training')

def run_Seq2Seq_Att_model():
    """
    该函数为测试Seq2Seq_Att模型的函数
    :return:
    """
    print("*"*50)

    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    print('加载数据...')
    X, y = read_data(config.dataroot, debug=False)
    print('运行已经跑好的模型')
    model = x.Model(
        X,
        y,
        config
    )

    model.load_state_dict(torch.load(config.save_model))
    print('==>模型加载成功！')
    model.test_model()

def main_Seq2Seq():
    """
    模型为Seq2Seq 模型的代码
    :return:
    """
    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()  # 加载模型的配置
    model = x.Seq2Seq(config)  # 加载模型
    print('==>当前使用的模型为：' + model_name)

    print('==>加载数据中...')
    # ntime_steps   为时间窗口T
    # n_next        为想要预测的天数
    dataset = MyDataset(ntime_steps=config.ntime_steps, n_next=config.n_next)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

    model.train(model, config, dataloader)  # Seq2Seq训练模型

    # model.test_model()  # 测试还有问题

def run_Seq2Seq_model():
    print("*"*100)

    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    print('运行已经跑好的模型')
    model = x.Seq2Seq(config)
    model.load_state_dict(torch.load(config.save_model))
    model.test_model()    # 测试有问题

def main_CNN_LSTM():
    """
    该函数为CNN_LSTM的模型函数
    :return:
    """
    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()

    model = x.TimeSeriesCNN_LSTM(config)

    print('==>当前使用的模型为：' + model_name)

    print('==>加载数据中...')
    # ntime_steps   为时间窗口T
    # n_next        为想要预测的天数
    dataset = MyDataset(ntime_steps=config.ntime_steps, n_next=config.n_next)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

    model.train(model, config, dataloader)

if __name__ == '__main__':


    main_Seq2Seq_Att()
    # main_Seq2Seq()
    #
    # run_Seq2Seq_model()  #有问题

    # main_CNN_LSTM()

    # m = nn.Linear(26,30)
    # hh = torch.randn(128,100,26)
    # hh2 = m(hh)
    # print(hh2.size())  # torch.Size([10, 128, 30])
















