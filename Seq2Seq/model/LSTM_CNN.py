#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/25/20 2:18 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : LSTM_CNN.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'LSTM_CNN'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size = 128
        self.test_batch_size = 100

        self.ntime_steps = 30  # 为时间窗口
        self.n_next = 7  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度
        self.hidden_dim = 128  # 隐藏层的大小

        self.dropout = 0.1
        self.num_layers = 1
        self.epochs = 3000
        self.lr = 0.001

        self.require_improvement = 300

        self.save_model = './data/check_point/best_LSTM_CNN_model_air_T:'+str(self.ntime_steps)+'_D:'+str(self.n_next)+'.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.window_sizes = [4, 5, 6]

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=config.input_size,
                                 hidden_size=config.hidden_dim,
                                 num_layers=config.num_layers,
                                 dropout=(0 if config.num_layers==1 else config.dropout),
                                 batch_first=True
                                 )

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim//2,
                    kernel_size=kernel_size
                ),
                nn.ReLU())
            # nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in config.window_sizes
        ])

        self.fc = nn.Linear(in_features=config.hidden_dim//2, out_features=config.n_next)
        self.dropout = nn.Dropout(config.dropout)

        self.fc2 = nn.Linear(in_features=18,out_features=1)

    def forward(self,inputs):
        outputs ,_= self.lstm(inputs)
        outputs = outputs.permute(0,2,1)

        outputs = [conv(outputs) for conv in self.convs]  # out[i]:batch_size x feature_size*1

        # output = [conv(x) for conv in self.convs]
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs.permute(0,2,1)

        # outputs = outputs.view(outputs.,-1)

        outputs = F.leaky_relu(self.dropout(self.fc(outputs)))

        outputs = outputs.permute(0,2,1)

        outputs = F.leaky_relu(self.dropout(self.fc2(outputs)))

        return outputs.squeeze(2)







