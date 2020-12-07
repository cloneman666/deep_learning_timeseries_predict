#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/7/20 8:26 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : MyDA.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Config(object):
    def __init__(self):
        self.model_name = 'DA_RNN'

        self.dataroot = './data/one_hot_甘.csv'

        self.batchsize = 128



        self.nhidden_encoder  = 128

        self.nhidden_decoder  = 128

        self.ntimestep  = 30   #时间窗口，即为T
        self.n_next = 1  # 为往后预测的天数

        self.save_model = './data/check_point/best_DA_RNN_air_T:'+str(self.ntimestep)+'_D:'+str(self.n_next)+'.pth'

        self.epochs  = 3000

        self.lr = 0.001

        self.require_improvement = 200   #超过100轮训练没有提升就结束训练


class Encoder(nn.Module):
    """encoder in Seq2Seq_Att."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in Seq2Seq_Att."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))


            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size),dim=-1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())
