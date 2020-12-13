#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/19/20 11:27 AM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : CNN_GRU.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'CNN_GRU'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size = 128
        self.test_batch_size = 100

        self.ntime_steps = 30  # 为时间窗口
        self.n_next = 15  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度
        self.hidden_dim = 100  # 隐藏层的大小

        self.num_layers = 1
        self.epochs = 3000
        self.lr = 0.001

        self.dropout = 0.1

        self.require_improvement = 1000

        self.save_model = './data/check_point/best_CNN_GRU_model_air_T:'+str(self.ntime_steps)+'D:'+ str(self.n_next)+'.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.window_sizes = [4, 5, 6]

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.input_size,
                    out_channels=config.hidden_dim,
                    kernel_size=kernel_size
                ),
                nn.ReLU())
            # nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in config.window_sizes
        ])

        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=config.num_layers,
            batch_first=True)

        self.fc = nn.Linear(in_features=config.hidden_dim//2,out_features=config.hidden_dim//4)
        self.fc2 = nn.Linear(in_features=config.hidden_dim//4,out_features=config.n_next)

        self.dropout = nn.Dropout(config.dropout)


    def forward(self,x):
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        x = x.permute(0, 2, 1)

        output = [conv(x) for conv in self.convs]  # out[i]:batch_size x feature_size*1

        # output = [conv(x) for conv in self.convs]
        output = torch.cat(output, dim=2)

        output = output.permute(0, 2, 1)

        output,_ = self.gru(output)

        # output = torch.tanh(self.fc(output[:,-1,:]))
        output = F.leaky_relu(self.fc(output[:,-1,:]))
        # output = F.dropout(output, p=0.1)
        output = F.leaky_relu(self.fc2(output))
        output = self.dropout(output)
        return output
