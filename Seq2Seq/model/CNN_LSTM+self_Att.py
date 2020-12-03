#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/3/20 3:30 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : CNN_LSTM+self_Att.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self):
        self.model_name = 'CNN_LSTM_self_Att'

        self.dataroot = './data/one_hot_甘.csv'


        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 2  # 为往后预测的天数

        self.save_model = './data/check_point/best_CNN_LSTM_self_Att_model_air_T:' + str(self.ntime_steps) + 'D:' + str(self.n_next) + '.pth'

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

        self.hid_size = 100
        # self.max_text_len = 20

        # self.num_class = 10

class Attention(nn.Module):
    def __init__(self,config):
        super(Attention, self).__init__()
        self.attn = nn.Linear(config.enc_hidden_dim + config.dec_hidden_dim ,config.dec_hidden_dim,bias=False)
        self.v = nn.Linear(config.dec_hidden_dim,1,bias=False)

    def forward(self,s,enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size,src_len, enc_hid_dim]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim]
        s = s.unsqueeze(1).repeat(1, src_len, 1)

        # enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.window_sizes = config.window_sizes

        ts_len = 20 # length of time series

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=20,
                    out_channels=config.hid_size,
                    kernel_size=kernel_size
                ),
                nn.ReLU())
                # nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in config.window_sizes
        ])
        # self.self_attention = Attention(config)

        self.lstm = nn.LSTM(
            input_size=config.hid_size,
            hidden_size=config.hid_size*2,
            num_layers=1,
            batch_first=True)

        self.fc = nn.Linear(in_features=config.hid_size*2,out_features=config.hid_size)
        self.fc2 = nn.Linear(in_features=config.hid_size,out_features=config.n_next)
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