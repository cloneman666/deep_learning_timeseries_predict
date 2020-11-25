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


class Seq2Seq(nn.Module):
    def __init__(self,config):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size=config.input_size,
                                 hidden_size=config.hidden_dim,
                                 num_layers=config.num_layers,
                                 dropout=(0 if config.num_layers==1 else config.dropout),
                                 batch_first=True
                                 )
        # self.decoder = nn.LSTM(input_size=)
