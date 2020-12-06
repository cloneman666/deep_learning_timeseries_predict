#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/6/20 7:46 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : DA_test.py
# @Software: PyCharm

import matplotlib.pyplot as plt

import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error,mean_squared_error
import time
import utils
import logging

class Config(object):
    def __init__(self):
        self.model_name = 'DA_test'

        self.dataroot = './data/one_hot_甘.csv'

        self.batchsize = 128



        self.nhidden_encoder  = 128

        self.nhidden_decoder  = 128

        self.ntimestep  = 30   #时间窗口，即为T
        self.n_next = 1  # 为往后预测的天数

        self.save_model = './data/check_point/best_DA_test_air_T:'+str(self.ntimestep)+'_D:'+str(self.n_next)+'.pth'

        self.epochs  = 3000

        self.lr = 0.001

        self.require_improvement = 200   #超过100轮训练没有提升就结束训练

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()

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