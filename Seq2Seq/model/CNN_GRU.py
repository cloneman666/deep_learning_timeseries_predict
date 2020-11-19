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

        self.batch_size = 128

        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 1  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度
        self.hidden_dim = 128  # 隐藏层的大小

        self.num_layers = 1
        self.epochs = 2000
        self.lr = 0.001

        self.require_improvement = 100

        self.save_model = './data/check_point/best_seq2seq_model_air.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.window_sizes = [3, 4, 5, 6]

class CNN_GRU(nn.Module):
    def __init__(self,config):
        super(CNN_GRU, self).__init__()


        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=20,
                    out_channels=config.hidden_dim,
                    kernel_size=kernel_size
                ),
                nn.ReLU())
            # nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in config.window_sizes
        ])

        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim//2,
            num_layers=config.num_layers,
            batch_first=True)

        self.fc = nn.Linear(in_features=config.hidden_dim//2,out_features=config.n_next)


    def forward(self,x):
        pass
