#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-11 19:53
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : Seq2Seq.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class Config(object):
    def __init__(self):
        self.model_name = 'Seq2Seq'

        self.input_size = 19   #输入数据的维度
        self.hidden_dim = 128  #隐藏层的大小

        self.num_layers = 1

        self.epochs  = 100

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.input_size = config.input_size
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Push through RNN layer (the ouput is irrelevant)
        _, self.hidden = self.lstm(inputs, self.hidden)
        return self.hidden


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        # input_size=1 since the output are single values
        self.lstm = nn.LSTM(1, hidden_size=config.hidden_dim, num_layers=config.num_layers)
        self.out = nn.Linear(config.hidden_dim, 1)

    def forward(self, outputs, hidden, criterion):
        batch_size, num_steps = outputs.shape
        # Create initial start value/token
        input = torch.tensor([[0.0]] * batch_size, dtype=torch.float)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        input = input.unsqueeze(0)

        loss = 0
        for i in range(num_steps):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            output = self.out(output[-1])
            # Generate input for next step by adding seq_len dimension (see above)
            input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
            loss += criterion(output, outputs[:, i])
        return loss


class Seq2Seq:
    def __init__(self,config):
        self.encoder = Encoder(config.input_size,config.hidden_dim)
        self.decoder = Decoder(config.hidden_dim)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.001)

        self.criterion = nn.MSELoss()

    def train(self,config):
        pass




