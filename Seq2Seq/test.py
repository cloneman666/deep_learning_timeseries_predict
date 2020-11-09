import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

from model.utils import *
from model.Seq2Seq_attention import *
from importlib import import_module

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="Seq2Seq+Att 进行时间序列预测")

    #选择模型即可
    parser.add_argument('--model_name',type=str,default='CNN_LSTM',help='choose a model Seq2Seq_attention、CNN_LSTM')

    # Dataset setting
    # parser.add_argument('--dataroot', type=str, default="./data/nasdaq/nasdaq100_padding.csv", help='path to dataset')
    # parser.add_argument('--dataroot', type=str, default="./data/one_hot_甘.csv", help='path to dataset')
    # parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')
    # parser.add_argument('--save_model',type=str,default="./data/check_point/best_model_air.pth",help='save the model')
    #
    # # Encoder / Decoder parameters setting
    # parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    # parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    # parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')
    #
    # # Training parameters setting
    # parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train [10, 200, 500]')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    #
    #
    # parser.add_argument('--require_improvement',type=int,default=100,help='默认100个epoch没有提升，就结束训练')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():

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

def run_model():
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

if __name__ == '__main__':
    # main()
    # run_model()

    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()
    print(config)







