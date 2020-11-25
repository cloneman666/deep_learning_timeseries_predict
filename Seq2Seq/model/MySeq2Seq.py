#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/25/20 2:18 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : MySeq2Seq.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.model_name = 'MySeq2Seq'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size = 128
        self.test_batch_size = 100

        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 1  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度
        self.hidden_dim = 128  # 隐藏层的大小

        self.dropout = 0.1
        self.num_layers = 1
        self.epochs = 2000
        self.lr = 0.001

        self.require_improvement = 100

        self.save_model = './data/check_point/best_seq2seq_model_air.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=config.input_size,
                                 hidden_size=config.hidden_dim,
                                 num_layers=config.num_layers,
                                 dropout=(0 if config.num_layers==1 else config.dropout),
                                 batch_first=True
                                 )
        self.fc = nn.Linear(in_features=config.hidden_dim,out_features=config.n_next)
        self.hidden = None

    def init_hidden(self, batch_size,config):
        return torch.zeros(self.num_layers,batch_size,self.hidden_dim).clone().detach().to(config.device),torch.zeros(self.num_layers,batch_size,self.hidden_dim).clone().detach().to(config.device)

    def forward(self,inputs):
        output, self.hidden = self.lstm(inputs, self.hidden)

        outputs = []
        for t in range(self.config.ntime_steps):
            outputs.append(self.fc(output[:,t,:]))

        return torch.stack(outputs,dim=1).squeeze(1)







