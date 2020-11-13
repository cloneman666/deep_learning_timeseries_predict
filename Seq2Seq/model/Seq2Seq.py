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
import utils
from model.utils import *
import time
import numpy as np
import matplotlib.pyplot as plt

class Config(object):
    def __init__(self):
        self.model_name = 'Seq2Seq'

        self.batch_size  = 128

        self.ntime_steps = 10 #为时间窗口
        self.n_next = 1       #为往后预测的天数

        self.input_size = 19   #输入数据的维度
        self.hidden_dim = 128  #隐藏层的大小

        self.num_layers = 1
        self.epochs  = 1000
        self.lr = 0.001

        self.require_improvement = 100

        self.save_model = './data/check_point/best_seq2seq_model_air.pth'

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.input_size = config.input_size
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_dim),
                torch.zeros(self.num_layers,batch_size,self.hidden_dim))

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


class Seq2Seq(nn.Module):
    def __init__(self,config):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr)

        self.criterion = nn.MSELoss()

    def train(self,model,config,dataloader):
        print('==>开始训练...')
        best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
        all_epoch = 0  # 记录进行了多少个epoch
        last_imporve = 0  # 记录上次校验集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升，停止训练

        start_time = time.time()

        for epoch in range(config.epochs):

            for i,train_data in enumerate(dataloader):

                train_x ,train_y= torch.tensor(train_data[0],dtype=torch.float),torch.tensor(train_data[1],dtype=torch.float)
                train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                train_y = train_y.squeeze(2)   #将最后的1去掉

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # Reset hidden state of encoder for current batch
                self.encoder.hidden = self.encoder.init_hidden(train_x.shape[1])
                # Do forward pass through encoder
                hidden = self.encoder(train_x)
                # Do forward pass through decoder (decoder gets hidden state from encoder)
                loss = self.decoder(train_y, hidden, self.criterion)
                # Backpropagation
                loss.backward()
                # Update parameters
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            # if epoch % 10 == 0:
            #     # y_train_pred = self.test(on_train=True)
            #     # y_test_pred = self.test(on_train=False)
            #     # y_pred = np.concatenate((y_train_pred, y_test_pred))
            #     hh1,hh2 = get_data(ntime_steps=config.ntime_steps, n_next=config.n_next)
            #
            #     plt.ion()
            #     plt.figure()
            #     plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
            #     plt.plot(range(self.T, len(y_train_pred) + self.T),
            #              y_train_pred, label='Predicted - Train')
            #     # plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
            #     #          y_test_pred, label='Predicted - Test')
            #     plt.legend(loc='upper left')
            #
            #     plt.pause(2)
            #     plt.close()

            if all_epoch % 10 == 0:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), config.save_model)
                    imporve = "*"
                    last_imporve = all_epoch
                else:
                    imporve = " "
                time_dif = utils.get_time_dif(start_time)

                msg = 'Epochs:{0:d},Loss:{1:.5f},Time:{2} {3}'

                print(msg.format(epoch,loss.item(), time_dif, imporve))

            all_epoch = all_epoch + 1

            if all_epoch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

            if flag:
                break



