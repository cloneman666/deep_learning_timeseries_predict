#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/6/20 7:30 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : GRU_Att.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Config(object):
    def __init__(self):
        self.model_name = 'GRU'

        self.dataroot = './data/one_hot_甘.csv'


        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 1  # 为往后预测的天数

        self.save_model = './data/check_point/best_GRU_model_air_T:' + str(self.ntime_steps) + 'D:' + str(self.n_next) + '.pth'

        self.hid_size = 100
        self.dropout = 0.1

        self.epochs = 3000  #需要的数比较大

        self.lr = 0.001

        self.require_improvement = 300  # 超过100轮训练没有提升就结束训练

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
            hidden_size=config.hid_size,
            num_layers=1,
            batch_first=True)

        self.fc = nn.Linear(in_features=config.hid_size*2,out_features=20)
        self.fc2 = nn.Linear(in_features=20,out_features=config.n_next)
        self.dropout = nn.Dropout(config.dropout)

        # x: [batch, seq_len, hidden_dim*2]
        # query : [batch, seq_len, hidden_dim * 2]
        # 软注意力机制 (key=value=x)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)

        return context, alpha_n

    def forward(self, x):

        output, _ = self.gru(x)

        query = self.dropout(output)

        attn_output, alpha = self.attention_net(output, query)

        output = F.leaky_relu((self.fc(attn_output)))

        output = F.leaky_relu(self.fc2(output))
        output = self.dropout(output)
        # output = F.softmax(output)
        return output