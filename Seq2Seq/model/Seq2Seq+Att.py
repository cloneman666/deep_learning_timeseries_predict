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

        self.ntime_steps = 10  # 为时间窗口
        self.n_next = 1  # 为往后预测的天数

        self.input_size = 20  # 输入数据的维度

        self.enc_hidden_dim = 128  # 隐藏层的大小
        self.dec_hidden_dim = 128

        self.dropout = 0.1
        self.num_layers = 1
        self.epochs = 2000
        self.lr = 0.001

        self.require_improvement = 100

        self.save_model = './data/check_point/best_seq2seq_model_air.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.enc_hidden_dim,
                            num_layers=config.num_layers,
                            dropout=(0 if config.num_layers==1 else config.dropout),
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=config.enc_hidden_dim,out_features=config.enc_hidden_dim//2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src):

        enc_output, enc_hidden = self.lstm(src)
        s = enc_hidden[-1,:,:]

        return enc_output,s

class Attention(nn.Module):
    def __init__(self,config):
        super(Attention, self).__init__()
        self.att = nn.Linear(config.enc_hidden_dim + config.dec_hidden_dim ,config.dec_hidden_dim,bias=False)
        self.v = nn.Linear(config.dec_hidden_dim,1,bias=False)

    def forward(self,s,enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()

        self.attention = Attention(config)

        self.lstm = nn.LSTM(input_size=config.enc_hidden_dim + config.input_size,
                            hidden_size=config.dec_hidden_dim
                            )
        self.fc_out = nn.Linear(in_features=config.enc_hidden_dim + config.dec_hidden_dim + config.input_size ,
                                out_features=config.n_next
                                )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,dec_input,s,enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1) #dec_input = [batch_size ,1]
        dec_input = dec_input.transpose(0,1)

        # a = [batch_size , 1 , src_len]
        a = self.attention(s,enc_output).unsqueeze(1)

        enc_output = enc_output.transpose(0,1)

        c = torch.bmm(a,enc_output).transpose(0,1)

        lstm_input = torch.cat((dec_input,c),dim=2)

        dec_output, dec_hidden = self.lstm(lstm_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, dec_input), dim=1))

        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,config):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(config).to(config.device)
        self.decoder = Decoder(config).to(config.device)

    def forward(self,src,trg,teacher_forcing_ratio = 0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return outputs


