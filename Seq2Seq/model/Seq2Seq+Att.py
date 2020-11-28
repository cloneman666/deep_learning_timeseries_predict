#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/24/20 6:17 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : Seq2Seq+Att.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Config(object):
    def __init__(self):
        self.model_name = 'Seq2Seq+Att'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size = 128
        self.test_batch_size = 100

        self.ntime_steps = 30  # 为时间窗口
        self.n_next = 7  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度

        self.enc_hidden_dim = 128  # 隐藏层的大小
        self.dec_hidden_dim = 128

        self.dropout = 0.1
        self.num_layers = 1
        self.epochs = 2000
        self.lr = 0.001

        self.require_improvement = 1000

        self.save_model = './data/check_point/best_Seq2Seq+Att_model_air_T:' + str(self.ntime_steps) + 'D:' + str(self.n_next) + '.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()

        self.gru = nn.GRU(input_size=config.input_size,
                            hidden_size=config.enc_hidden_dim,
                            num_layers=config.num_layers,
                            dropout=(0 if config.num_layers==1 else config.dropout),
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=config.enc_hidden_dim,out_features=config.enc_hidden_dim//2)

    def forward(self, src):

        enc_output, enc_hidden = self.gru(src)

        s = enc_hidden[-1,:,:]  #为单层，传最后一层即可

        return enc_output,s

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


class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.n_next = config.n_next
        self.attention = Attention(config)

        self.gru = nn.GRU(input_size=config.enc_hidden_dim + config.n_next,
                          hidden_size=config.dec_hidden_dim,
                          num_layers=config.num_layers,
                          dropout=(0 if config.num_layers == 1 else config.dropout),
                          batch_first=True
                            )
        self.fc_out = nn.Linear(in_features=config.enc_hidden_dim + config.dec_hidden_dim + config.n_next ,
                                out_features=config.n_next
                                )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,dec_input,s,enc_output):  #dec_input 此时输入的为y
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim]

        #dec_input = [batch_size ,1]
        dec_input = dec_input.unsqueeze(1)


        # a = [batch_size , 1 , src_len]
        a = self.attention(s,enc_output).unsqueeze(1)

        c = torch.bmm(a,enc_output)

        gru_input = torch.cat((dec_input,c),dim=2)

        dec_output, dec_hidden = self.gru(gru_input, s.unsqueeze(0))

        # pred = [batch_size, output_dim]

        pred = self.fc_out(torch.cat((dec_output, c, dec_input), dim=2))
        pred = pred.permute(0,2,1)

        return pred,dec_hidden.squeeze(0)


class Model(nn.Module):
    """
    该类为Seq2Seq+Attention模型
    """
    def __init__(self,config):
        super(Model, self).__init__()
        self.device = config.device
        self.encoder = Encoder(config).to(config.device)
        self.decoder = Decoder(config).to(config.device)
        self.attention = Attention(config).to(config.device)

    def forward(self,x,trg,teacher_forcing_ratio = 0.5): #dec_input就为y

        batch_size = x.shape[0]
        # trg_len = trg.shape[1]
        n_next = self.decoder.n_next   #往后预测的天数

        # tensor to store decoder outputs   作为存储decoder层的输出  类似（128，1）
        # outputs = torch.zeros(batch_size, n_next) #.to(self.device)

        # outputs = []

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(x)


        for t in range(n_next):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(trg,s, enc_output)  #dec_input

            # hh = dec_output[-1]  #最后一层的数据

            # place predictions in a tensor holding predictions for each token
            # outputs[t] = dec_output[:,-1,:]

            # output = dec_output[:,-1,:]

            # decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            # top1 = dec_output.argmax(2)

            trg = dec_output.squeeze(2) #if teacher_force else top1

            # outputs.append(dec_output)

        # outputs = torch.cat(outputs,dim=1)

        # outputs = torch.stack(outputs)
        return dec_output.squeeze(2)


