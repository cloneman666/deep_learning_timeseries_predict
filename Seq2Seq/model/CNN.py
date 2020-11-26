#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/23/20 9:47 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : sample_CNN_LSTM.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'CNN'

        self.dataroot = './data/one_hot_甘.csv'

        self.save_model = './data/check_point/best_CNN_model_air_T:'+self.ntime_steps+'D:'+ self.n_next+'.pth'

        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 1  # 为往后预测的天数

        self.dropout = 0.1

        self.epochs = 3000

        self.lr = 0.001

        self.require_improvement = 1000  # 超过300轮训练没有提升就结束训练

        self.batch_size = 128
        self.test_batch_size = 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.feature_size = 20

        self.window_sizes = [4,5,6]

        self.max_text_len = 20


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

        self.fc = nn.Linear(in_features=hid_size *18,out_features=hid_size*18//4)
        self.fc2 = nn.Linear(in_features=hid_size*18//4,out_features=config.n_next)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        x = x.permute(0,2,1)

        output = [conv(x) for conv in self.convs]  # out[i]:batch_size x feature_size*1

        # output = [conv(x) for conv in self.convs]
        output = torch.cat(output, dim=2)

        output = output.view(-1,output.size(1)*output.size(2))

        # output = output.permute(0,2,1)
        # output = output.view(output.size(0) * output.size(1) , -1)

        # output = F.relu(self.fc(output))

        # output, (h_n, h_c) = self.lstm(output)

        # output = F.leaky_relu((self.fc(output[:, -1, :])))

        output = self.dropout(F.leaky_relu(self.fc(output)))
        output = self.dropout(F.leaky_relu(self.fc2(output)))
        output = self.dropout(output)
        # output = F.softmax(output)

        return output

