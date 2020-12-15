#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 11:21
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : Seq2Seq_attention.py
# @Software: PyCharm

# 此模型主要为预测金融时间序列预测提出的，加入了注意力机制
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
from model.utils import *

class Config(object):
    def __init__(self):
        self.model_name = 'DA_RNN'

        self.dataroot = './data/one_hot_甘.csv'

        self.batchsize = 128



        self.nhidden_encoder  = 128

        self.nhidden_decoder  = 128

        self.ntime_steps  = 25   #时间窗口，即为T
        self.n_next = 1  # 为往后预测的天数，目前此版本只能为1

        self.save_model = './data/check_point/best_DA_RNN_air_T:'+str(self.ntime_steps)+'_D:'+str(self.n_next)+'.pth'

        self.epochs  = 3000

        self.lr = 0.001

        self.require_improvement = 200   #超过100轮训练没有提升就结束训练

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class Decoder(nn.Module):
    """decoder in Seq2Seq_Att."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden,n_next):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        self.n_next = n_next

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, self.n_next)

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

    def __init__(self, X, y, config,
                 parallel=False):
        """initialization."""
        super(Model, self).__init__()
        self.encoder_num_hidden = config.nhidden_encoder
        self.decoder_num_hidden = config.nhidden_decoder
        self.learning_rate = config.lr
        self.batch_size = config.batchsize
        self.parallel = parallel
        self.shuffle = False
        self.epochs = config.epochs
        self.T = config.ntime_steps
        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=config.nhidden_encoder,
                               T=config.ntime_steps).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=config.nhidden_encoder,
                               decoder_num_hidden=config.nhidden_decoder,
                               T=config.ntime_steps,n_next=config.n_next).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[:self.train_timesteps])
        self.input_size = self.X.shape[1]

    def train(self,model,config):  #参数model用于将模型保存时用到，config 用于存储模型位置的时候用到

        logging_name = config.model_name + '_D:' + str(config.ntime_steps) + '_D:'+str(config.n_next)
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', filename='./log/' + logging_name + '.log',
                            level=logging.INFO)

        """Training process."""
        iter_per_epoch = int(
            np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        best_loss = float('inf') #记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
        all_epoch = 0  #记录进行了多少个epoch
        last_imporve = 0  # 记录上次校验集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升，停止训练

        start_time = time.time()
        for epoch in tqdm(range(self.epochs)):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(
                        indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]

                y_predicted, y_true,loss = self.train_forward(x, y_prev, y_gt)

                self.iter_losses[int(
                    epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])


            # if epoch % 10 == 0:
            #     y_train_pred = self.test(on_train=True)
            #     y_test_pred = self.test(on_train=False)
            #     y_pred = np.concatenate((y_train_pred, y_test_pred))
            #     plt.ion()
            #     plt.figure()
            #     plt.plot(range(1, 1 + len(self.y)), self.y, label="Ground Truth")
            #     plt.plot(range(self.T, len(y_train_pred) + self.T),
            #              y_train_pred, label='Predicted - Train')
            #     plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
            #              y_test_pred, label='Predicted - Test')
            #     plt.legend(loc='upper left')
            #     plt.pause(2)
            #     plt.close()
            #     # plt.draw()
                # plt.show()

            if all_epoch % 10 == 0:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(),config.save_model)
                    imporve = "*"
                    last_imporve = all_epoch
                else:
                    imporve = " "
                time_dif = utils.get_time_dif(start_time)

                ###############此处的评估需要增加#############

                MSE, RMSE, MAE = evaluation(y_predicted.detach().cpu().numpy(),y_true.cpu().numpy())

                msg = 'Epochs:{0:d},Iterations:{1:d}, Loss:{2:.5f}, MSE:{3:.5f}, RMSE:{4:.5f}, MAE:{5:.5f}, Time:{6} {7}'

                print(msg.format(epoch, n_iter,self.epoch_losses[epoch], MSE, RMSE, MAE, time_dif, imporve))
                logging.info(msg.format(epoch, n_iter,self.epoch_losses[epoch], MSE, RMSE, MAE, time_dif, imporve))

                ###########################################

                # msg = 'Epochs:{0:d},Iterations:{1:d},Loss:{2:.5f},Time:{3} {4}'
                #
                # print(msg.format(epoch,n_iter,self.epoch_losses[epoch],time_dif,imporve))
                # print("Epochs: ", epoch, " Iterations: ", n_iter,
                #       " Loss: ", self.epoch_losses[epoch] + str(imporve))
            all_epoch = all_epoch + 1

            if all_epoch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

            if flag:
                break


    def train_forward(self, X, y_prev, y_gt):
        """Forward pass."""
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))

        y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return y_pred,y_true ,loss.item()

    def test(self, on_train=False):
        """Prediction."""
        pic_data = read_all_data('./data/one_hot_甘.csv')

        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:

                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]

                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(self.device))

            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))

            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,y_history).cpu().data.numpy()[:, 0]

            i += self.batch_size

        return y_pred,pic_data

    def test_model(self,config):

        y_train_pred ,pic_data= self.test(on_train=True)
        y_test_pred,_ = self.test(on_train=False)
        y_pred = np.concatenate((y_train_pred, y_test_pred))

        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.set_title(config.model_name + '_T:' + str(config.ntime_steps) + '_D:' + str(config.n_next))
        # ax0.plot(range(1, 1 + len(Y1)), Y1, label='Ground Truth')
        ax0.plot(pic_data.ds, pic_data.y, label='Ground Truth')
        ax0.plot(pic_data.ds[config.ntime_steps:config.ntime_steps + len(y_train_pred)], y_train_pred, alpha=0.5,
                 label='Predicted - Train')
        ax0.plot(pic_data.ds[config.ntime_steps + len(y_train_pred) - 1:], y_test_pred, alpha=0.5,
                 label='Predicted - Test')

        ax1.set_title('Magnifies test set predictions')
        ax1.plot(pic_data.ds[config.ntime_steps + len(y_train_pred) - 1:],
                 pic_data.y[config.ntime_steps + len(y_train_pred) - 1:], label='Ground Truth')
        ax1.plot(pic_data.ds[config.ntime_steps + len(y_train_pred) - 1:], y_test_pred, 'g', alpha=0.5,
                 label='Predicted - Test')

        for tick in ax1.get_xticklabels():
            tick.set_rotation(25)

        # plt.ioff()
        # plt.figure(figsize=(10,3),dpi=300)
        # plt.title('DA_RNN_T:'+str(config.ntimestep)+'D:1')
        # plt.plot(range(1, 1 + len(self.y)), self.y, label="Ground Truth")
        #
        # plt.plot(range(self.T, len(y_train_pred) + self.T),y_train_pred, label='Predicted - Train')
        # plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),y_test_pred, label='Predicted - Test')
        # plt.tight_layout()
        # plt.legend(loc='upper left')
        # plt.savefig('./data/pic/DA_RNN_T:'+str(config.ntimestep)+'D:1.png')
        ax0.legend()
        ax1.legend()
        plt.tight_layout()
        plt.savefig('./data/pic/DA_RNN_T:' + str(config.ntime_steps) + 'D:1.png')
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
