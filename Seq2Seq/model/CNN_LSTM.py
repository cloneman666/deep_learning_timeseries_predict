#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 11:15
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : CNN_LSTM.py
# @Software: PyCharm


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model.utils import *
import utils

import torch.optim as optim
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


class Config(object):
    def __init__(self):
        self.model_name = 'CNN_LSTM'

        self.dataroot = './data/one_hot_甘.csv'


        self.ntime_steps = 30  # 为时间窗口
        self.n_next = 7  # 为往后预测的天数

        self.save_model = './data/check_point/best_CNN_LSTM_model_air_T:' + str(self.ntime_steps) + 'D:' + str(self.n_next) + '.pth'

        self.dropout = 0.1

        self.epochs = 3000

        self.lr = 0.001

        self.require_improvement = 1000  # 超过100轮训练没有提升就结束训练

        self.batch_size = 128
        self.test_batch_size = 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.feature_size = 20

        self.window_sizes = [4,5,6]

        self.max_text_len = 20

        # self.num_class = 10

# https://blog.csdn.net/sunny_xsc1994/article/details/82969867

# https://discuss.pytorch.org/t/multi-step-time-series-lstm-network/41670/6


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.window_sizes = config.window_sizes

        ts_len = 20 # length of time series
        hid_size = 100
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=20,
                    out_channels=hid_size,
                    kernel_size=kernel_size
                ),
                nn.ReLU())
                # nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in config.window_sizes
        ])

        self.lstm = nn.LSTM(
            input_size=hid_size,
            hidden_size=hid_size*2,
            num_layers=1,
            batch_first=True)

        self.fc = nn.Linear(in_features=hid_size*2,out_features=hid_size)
        self.fc2 = nn.Linear(in_features=hid_size,out_features=config.n_next)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        x = x.permute(0,2,1)

        output = [conv(x) for conv in self.convs]  # out[i]:batch_size x feature_size*1

        # output = [conv(x) for conv in self.convs]
        output = torch.cat(output, dim=2)


        output = output.permute(0,2,1)
        # output = output.view(output.size(0) * output.size(1) , -1)

        # output = F.relu(self.fc(output))

        output, (h_n, h_c) = self.lstm(output)

        output = F.leaky_relu((self.fc(output[:, -1, :])))

        output = F.leaky_relu(self.fc2(output))
        output = self.dropout(output)
        # output = F.softmax(output)
        return output

# def evaluation(y_true, y_pred):
#         """
#         该函数为计算，均方误差、平方绝对误差、平方根误差
#         :param y_true:
#         :param y_pred:
#         :return:
#         """
#     MSE = mean_squared_error(y_true, y_pred)  # 均方误差
#
#     MAE = mean_absolute_error(y_true, y_pred)  # 平方绝对误差
#
#     RMSE = np.sqrt(mean_squared_error(y_true, y_pred))  # 此为均方误差的开平方
#     return MSE, RMSE, MAE


    # def train(self,model, config, dataloader):
    #     self.criterion = nn.MSELoss()
    #     self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
    #     print('==>开始训练...')
    #     best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
    #     all_epoch = 0  # 记录进行了多少个epoch
    #     last_imporve = 0  # 记录上次校验集loss下降的batch数
    #     flag = False  # 记录是否很久没有效果提升，停止训练
    #
    #     start_time = time.time()
    #
    #     for epoch in range(config.epochs):
    #
    #         for i, train_data in enumerate(dataloader):
    #             train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32), torch.as_tensor(train_data[1],dtype=torch.float32)
    #             # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
    #             train_y = train_y.squeeze(2)  # 将最后的1去掉
    #
    #             self.optimizer.zero_grad()
    #
    #             output = model(train_x)
    #             loss = self.criterion(output,train_y)
    #             loss.backward()
    #             self.optimizer.step()
    #
    #         if epoch % 10 ==0:
    #             plt.ion()
    #             plt.figure()
    #             plt.plot(range(1, 1 + len(train_y)), train_y, label="True")
    #             plt.plot(range(1, 1 + len(output.detach().numpy())), output.detach().numpy(), label="Test")
    #             plt.legend()
    #             plt.pause(1)
    #             plt.close()
    #
    #
    #         if all_epoch % 10 == 0:
    #             if loss < best_loss:
    #                 best_loss = loss
    #                 torch.save(model.state_dict(), config.save_model)
    #                 imporve = "*"
    #                 last_imporve = all_epoch
    #             else:
    #                 imporve = " "
    #             time_dif = utils.get_time_dif(start_time)
    #
    #             MSE, RMSE, MAE = evaluation(train_y, output.detach().numpy())
    #
    #             msg = 'Epochs:{0:d}, Loss:{1:.5f}, MSE:{2:.5f}, RMSE:{3:.5f}, MAE:{4:.5f}, Time:{5} {6}'
    #
    #             print(msg.format(epoch, loss.item(), MSE, RMSE, MAE, time_dif, imporve))
    #
    #             # msg = 'Epochs:{0:d},Loss:{1:.5f},Time:{2} {3}'
    #             #
    #             # print(msg.format(epoch,loss.item(), time_dif, imporve))
    #
    #         all_epoch = all_epoch + 1
    #
    #         if all_epoch - last_imporve > config.require_improvement:
    #             # 在验证集合上loss超过1000batch没有下降，结束训练
    #             print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
    #             flag = True
    #             break
    #
    #         if flag:
    #             break





# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(19, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         # x = F.relu(self.fc1(x))
#         # x = F.dropout(x, training=self.training)
#         # x = self.fc2(x)
#         # return F.log_softmax(x, dim=1)
#         return x


# class Test_Model(nn.Module):
#     def __init__(self):
#         super(Test_Model, self).__init__()
#         self.cnn = CNN()
#         self.rnn = nn.LSTM(
#             input_size=320,
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True)
#         self.linear = nn.Linear(64, 3)
#
#     def forward(self, x):
#         batch_size, timesteps, C, H, W = x.size()
#         c_in = x.view(batch_size * timesteps, C, H, W)
#         c_out = self.cnn(c_in)
#         r_in = c_out.view(batch_size, timesteps, -1)
#         r_out, (h_n, h_c) = self.rnn(r_in)
#         r_out2 = self.linear(r_out[:, -1, :])
#
#         return F.log_softmax(r_out2, dim=1)
