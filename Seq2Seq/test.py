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
    parser.add_argument('--model_name',type=str,default='Seq2Seq',help='choose a model Seq2Seq、Seq2Seq_attention、CNN_LSTM')

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

    #####        main 和run_model 两个函数为模型：Seq2Seq_attention
    # main()
    # run_model()
    #
    args = parse_args()  # 加载所选模型的名字
    model_name = args.model_name
    x = import_module('model.' + model_name)
    config = x.Config()  #加载模型的配置
    model = x.Seq2Seq(config)   #加载模型
    print('==>当前使用的模型为：'+ model_name)

    print('==>加载数据中...')
    # ntime_steps   为时间窗口T
    # n_next        为想要预测的天数
    dataset = MyDataset(ntime_steps=config.ntime_steps,n_next=config.n_next)
    dataloader = DataLoader(dataset=dataset,batch_size=config.batch_size,shuffle=False)


    model.train(model,config,dataloader)

    # print("*"*50)
    # inputs = [
    #     [
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #     ],
    #     [
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #         [0.5, 0.2, 0.3, 0.4, 0.1],
    #     ]
    # ]
    #
    # inputs = torch.tensor(np.array(inputs), dtype=torch.float)
    # # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
    # inputs = inputs.transpose(1, 0)
    # print(inputs.shape)

    # outputs = [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.3, 0.2, 0.1]]
    # outputs = torch.tensor(np.array(outputs), dtype=torch.float)
    # print(outputs.shape)

    # optimizer = optim.Adam(model.parameters(),lr=0.001)
    #
    # criterion = nn.MSELoss()
    #
    # num_epochs = 100
    #
    # # Read dataset
    # print("==> Load dataset ...")
    # # ntime_steps   为时间窗口
    # # n_next        为想要预测的天数
    # dataset = MyDataset(ntime_steps=10,n_next=3)
    #
    # dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=False)
    #
    # for epoch in range(num_epochs):
    #     print(f'Starting epoch {epoch+1}/{num_epochs}')
    #
    #     model.train()
    #     for i,train_data in enumerate(dataloader):
    #         # print(i)
    #         # print(type(train_data[0]),type(train_data[1]))
    #
    #
    #         train_x ,train_y= train_data[0],train_data[1]
    #
    #         # print(train_x.float(),train_y.float())
    #
    #         output = model(train_x.float())
    #
    #         loss = criterion(output,train_y)
    #     # print(train_x,train_y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()











