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
from sklearn.metrics import mean_absolute_error,mean_squared_error
from torch import optim
import matplotlib.pyplot as plt
import time
import utils
import numpy as np
from model.utils import *

class Config(object):
    def __init__(self):
        self.model_name = 'MyDA'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size = 128
        self.test_batch_size = 100

        self.nhidden_encoder  = 128

        self.nhidden_decoder  = 128

        self.ntime_steps  = 30   #时间窗口，即为T
        self.n_next = 7  # 为往后预测的天数

        self.save_model = './data/check_point/best_MyDA_air_T:'+str(self.ntime_steps)+'_D:'+str(self.n_next)+'.pth'

        self.epochs  = 3000

        self.lr = 0.001

        self.require_improvement = 200   #超过100轮训练没有提升就结束训练

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

class Encoder(nn.Module):
    """encoder in Seq2Seq_Att."""

    def __init__(self, config):
        """Initialize an encoder in Seq2Seq_Att."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = config.nhidden_encoder
        self.input_size = 20

        self.T = config.ntime_steps

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1,
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T ,
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

            x = self.encoder_attn(x.view(-1, self.encoder_num_hidden * 2 + self.T ))

            # x = self.encoder_attn(x)

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size),dim=-1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))

            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())

class Decoder(nn.Module):
    """decoder in Seq2Seq_Att."""

    def __init__(self, config):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = config.nhidden_decoder
        self.encoder_num_hidden = config.nhidden_encoder
        self.T = config.ntime_steps
        self.n_next = config.n_next

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * self.decoder_num_hidden +
                      self.encoder_num_hidden, self.encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(self.encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.decoder_num_hidden
        )
        self.fc = nn.Linear(self.encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(self.decoder_num_hidden + self.encoder_num_hidden, self.n_next)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1),dim=-1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()

                _, final_states = self.lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())

class Model(nn.Module):
    """ Seq2Seq_Att Recurrent Neural Network."""

    def __init__(self,config):
        """initialization."""
        super(Model, self).__init__()
        self.encoder_num_hidden = config.nhidden_encoder
        self.decoder_num_hidden = config.nhidden_decoder

        self.device = config.device
        self.Encoder = Encoder(config).to(config.device)
        self.Decoder = Decoder(config).to(config.device)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=config.lr)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=config.lr)
        # Loss function
        self.criterion = nn.MSELoss()

    def train_forward(self, train_x, y_prev, y_gt):
        """Forward pass."""
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(train_x.to(self.device))

        y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        # y_true = Variable(y_gt)

        # y_true = y_gt.view(-1, 1)
        loss = self.criterion(y_pred, y_gt)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return y_pred,y_gt ,loss.item()

    def train(self,model,config,train_dataloader,test_dataloader):

        best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
        last_imporve = 0  # 记录上次校验集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升，停止训练
        start_time = time.time()
        for epoch in range(config.epochs):

            for i, train_data in enumerate(train_dataloader):
                train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32).to(
                    config.device), torch.as_tensor(train_data[1], dtype=torch.float32).to(config.device)
                # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                train_y = train_y.squeeze(2)  # 将最后的1去掉
                # hh = train_x.size(0)
                y_prev = np.zeros((train_x.size(0), config.ntime_steps - 1))  #batch_size  要能整除

                y_pred, y_true, loss = self.train_forward(train_x,y_prev,train_y)

            if epoch % 10 == 0:
                y_train_pred ,Y1= self.draw_pic(model,config,on_train=True)
                y_test_pred,Y1 = self.draw_pic(model,config,on_train=False)
                y_predhh = np.concatenate((y_train_pred, y_test_pred))
                plt.ion()
                plt.figure()
                plt.plot(range(1, 1 + len(Y1)), Y1, label="Ground Truth")

                plt.plot(range(config.ntime_steps, len(y_train_pred) + config.ntime_steps),y_train_pred, label='Predicted - Train')

                plt.plot(range(config.ntime_steps + len(y_train_pred), len(Y1) + 1),y_test_pred, label='Predicted - Test')

                plt.legend(loc='upper left')
                plt.pause(2)
                plt.close()
                # plt.draw()
                # plt.show()

            if epoch % 10 == 0:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), config.save_model)
                    imporve = "*"
                    last_imporve = epoch
                else:
                    imporve = " "
                time_dif = utils.get_time_dif(start_time)

                MSE, RMSE, MAE = evaluation(y_pred.detach().cpu().numpy(), y_true.cpu().numpy())
                msg = 'Epochs:{0:d},Loss:{1:.5f}, MSE:{2:.5f}, RMSE:{3:.5f}, MAE:{4:.5f}, Time:{5} {6}'
                print(msg.format(epoch, loss, MSE, RMSE, MAE, time_dif, imporve))

            if epoch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

            if flag:
                break

    def draw_pic(self,model,config,on_train=True):
        """
        此函数为画图函数，将每步的图像进行可视化
        """
        X1, Y1 = read_data(config.dataroot)
        train_timesteps = int(X1.shape[0] * 0.8)

        if on_train:
            y_pred = np.zeros(train_timesteps - config.ntime_steps + 1)
        else:
            y_pred = np.zeros(X1.shape[0] - train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + config.batch_size)]
            X = np.zeros((len(batch_idx), config.ntime_steps , X1.shape[1]))
            y_history = np.zeros((len(batch_idx), config.ntime_steps ))

            for j in range(len(batch_idx)):
                if on_train:

                    X[j, :, :] = X1[range(batch_idx[j], batch_idx[j] + config.ntime_steps ), :]

                    y_history[j, :] = Y1[range(batch_idx[j], batch_idx[j] + config.ntime_steps )]

                else:
                    X[j, :, :] = X1[range(batch_idx[j] + train_timesteps - config.ntime_steps,
                                          batch_idx[j] + train_timesteps ), :]

                    y_history[j, :] = Y1[
                        range(batch_idx[j] + train_timesteps - config.ntime_steps, batch_idx[j] + train_timesteps)]

            # 这里用不到
            y_history = torch.from_numpy(y_history).type(torch.FloatTensor).to(config.device)

            # train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32), torch.as_tensor(train_data[1],
            #                                                                                         dtype=torch.float32)
            # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
            # train_y = train_y.squeeze(2)  # 将最后的1去掉

            _, input_encoded = self.Encoder((torch.from_numpy(X).type(torch.FloatTensor).to(config.device)))

            y_pred[i:(i + config.batch_size)] = self.Decoder(input_encoded,y_history).cpu().data.numpy()[:, 0]

            # y_pred[i:(i + config.batch_size)] = model(
            #     torch.as_tensor(X, dtype=torch.float32).to(config.device)).detach().cpu().numpy()[:, 0]

            i += config.batch_size

        return y_pred,Y1

    def test(self,model, config):
        print("*" * 100)
        print("==>加载已训练好的模型...")

        model.load_state_dict(torch.load(config.save_model, map_location=torch.device(config.device)))

        y_train_pred, Y1 = self.draw_pic(model, config, on_train=True)
        y_test_pred, Y1 = self.draw_pic(model, config, on_train=False)

        plt.ion()

        plt.figure(figsize=(10, 3), dpi=300)
        plt.title(config.model_name + '_T:' + str(config.ntime_steps) + '_D:' + str(config.n_next))
        plt.plot(range(1, 1 + len(Y1)), Y1, label='Ground Truth')
        plt.plot(range(config.ntime_steps + len(y_train_pred), len(Y1) + 1), y_test_pred, label='Predicted - Test')

        plt.plot(range(config.ntime_steps, len(y_train_pred) + config.ntime_steps), y_train_pred,
                 label='Predicted - Train')
        plt.tight_layout()
        plt.legend()
        # plt.pause(1)
        # plt.close()
        plt.savefig(
            './data/pic/' + config.model_name + '_T:' + str(config.ntime_steps) + '_D:' + str(config.n_next) + '.png')
        plt.show()



def evaluation(y_true, y_pred):
    """
    该函数为计算，均方误差、平方绝对误差、平方根误差
    :param y_true:
    :param y_pred:
    :return:
    """
    MSE = mean_squared_error(y_true,y_pred)  #均方误差

    MAE = mean_absolute_error(y_true,y_pred)  #平方绝对误差

    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))  #此为均方误差的开平方
    return MSE,RMSE,MAE