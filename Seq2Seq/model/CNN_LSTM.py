#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 11:15
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : CNN_LSTM.py
# @Software: PyCharm


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


class Config(object):
    def __init__(self):
        self.model_name = 'CNN_LSTM'

        self.dataroot = './data/one_hot_甘.csv'

        self.save_model = './data/check_point/best_model_air.pth'

        self.nhidden_encoder = 128

        self.nhidden_decoder = 128

        self.ntimestep = 10  # 时间窗口，即为T

        self.epochs = 1000

        self.lr = 0.001

        self.require_improvement = 100  # 超过100轮训练没有提升就结束训练

        self.batch_size = 50

        self.test_batch_size = 1000

        self.momentum = 0.5

        self.log_interval = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_size = 20

        self.window_sizes = [3,4,5,6]

        self.max_text_len = 20

        self.num_class = 10

# https://blog.csdn.net/sunny_xsc1994/article/details/82969867

# https://discuss.pytorch.org/t/multi-step-time-series-lstm-network/41670/6


class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        kernel_sizes = [4, 5, 6]
        ts_len = 19 # length of time series
        hid_size = 100
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=19,
                    out_channels=10,
                    kernel_size=kernel_size,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=ts_len - kernel_size + 1))
            for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(
            in_features=hid_size * len(kernel_sizes),
            out_features=3,
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        output = [conv(x) for conv in self.convs]
        output = torch.cat(output, dim=1)
        output = output.view(output.size(1), -1)
        output = self.fc(output)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(19, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=320,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64, 3)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)


# model = Combine()
# if args.cuda:
#     model.cuda()
#
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#
#         data = np.expand_dims(data, axis=1)
#         data = torch.FloatTensor(data)
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#
#
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#
#         data = np.expand_dims(data, axis=1)
#         data = torch.FloatTensor(data)
#         print(target.size)
#
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         # with torch.no_grad:
#
#         data, target = Variable(data,volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
#
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
#
#         test_loss /= len(test_loader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#                 test_loss, correct, len(test_loader.dataset),
#                 100. * correct / len(test_loader.dataset)))
#
#
# for epoch in range(1, args.epochs + 1):
#     train(epoch)
#     test()