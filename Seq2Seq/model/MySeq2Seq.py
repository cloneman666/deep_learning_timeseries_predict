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
        self.epochs = 3000
        self.lr = 0.001

        self.require_improvement = 300

        self.save_model = './data/check_point/best_my_seq2seq_model_air.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.gru = nn.GRU(input_size=config.input_size,
                                 hidden_size=config.hidden_dim,
                                 num_layers=config.num_layers,
                                 dropout=(0 if config.num_layers==1 else config.dropout),
                                 batch_first=True
                                 )

        self.fc1 = nn.Linear(in_features=config.hidden_dim , out_features=1)



    def attention_net(self,x):  #x:[batch, seq_len, hidden_dim]

        u = torch.tanh(torch.matmul(x,self.w_omega))  #[batch, seq_len, hidden_dim]
        att = torch.matmul(u,self.u_omega)            #[batch, seq_len, 1]
        att_score = F.softmax(att,dim=1)

        score_x = x * att_score

        context = torch.sum(score_x,dim=1)
        return context

    def forward(self,inputs):
        output, _ = self.lstm(inputs)
        output = output.permute(1,0,2)

        attn_output = self.attention_net(output)
        logit = self.fc1(attn_output)


        return logit







