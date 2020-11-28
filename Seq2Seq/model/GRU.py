#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/23/20 11:36 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : GRU.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'GRU'

        self.dataroot = './data/one_hot_甘.csv'


        self.ntime_steps = 30  # 为时间窗口
        self.n_next = 7  # 为往后预测的天数

        self.save_model = './data/check_point/best_GRU_model_air_T:' + str(self.ntime_steps) + 'D:' + str(self.n_next) + '.pth'

        self.dropout = 0.1

        self.epochs = 3000

        self.lr = 0.001

        self.require_improvement = 1000  # 超过100轮训练没有提升就结束训练

        self.batch_size = 128
        self.test_batch_size = 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.feature_size = 20

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()

        self.gru = nn.GRU(
            input_size=20,
            hidden_size=20*2,
            num_layers=1,
            batch_first=True)

        self.fc = nn.Linear(in_features=20*2,out_features=20)
        self.fc2 = nn.Linear(in_features=20,out_features=config.n_next)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        output, _ = self.gru(x)

        output = F.leaky_relu((self.fc(output[:, -1, :])))

        output = F.leaky_relu(self.fc2(output))
        output = self.dropout(output)
        # output = F.softmax(output)
        return output