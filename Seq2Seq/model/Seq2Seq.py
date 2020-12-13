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
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import logging

class Config(object):
    def __init__(self):
        self.model_name = 'Seq2Seq'

        self.dataroot = './data/one_hot_甘.csv'

        self.batch_size  = 128

        self.ntime_steps = 30 #为时间窗口
        self.n_next = 3       #为往后预测的天数

        self.input_size = 20   #输入数据的维度
        self.hidden_dim = 128  #隐藏层的大小

        self.num_layers = 1
        self.epochs  = 2000
        self.lr = 0.001

        self.require_improvement = 200

        self.save_model = './data/check_point/best_seq2seq_model_air_T:'+str(self.ntime_steps)+'_D:'+ str(self.n_next)+'.pth'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.input_size = config.input_size
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers,batch_first=True)
        self.hidden = None

    def init_hidden(self, batch_size,config):
        return torch.zeros(self.num_layers,batch_size,self.hidden_dim).clone().detach().to(config.device),torch.zeros(self.num_layers,batch_size,self.hidden_dim).clone().detach().to(config.device)

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

    def forward(self, outputs, hidden, criterion,config):
        batch_size, num_steps = outputs.shape
        # Create initial start value/token
        input = torch.tensor([[0.0]] * batch_size, dtype=torch.float32)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        input = input.unsqueeze(0)
        input = input.to(config.device)
        loss = 0
        outs = []
        for i in range(num_steps):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            output = self.out(output[-1])
            # Generate input for next step by adding seq_len dimension (see above)
            input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
            # output.squeeze(1)  去除警告，将特定维度降低
            output = output.squeeze(1)
            loss += criterion(output, outputs[:, i])
            outs.append(output)

        hh = torch.stack(outs,dim=1)

        return torch.stack(outs,dim=1),loss


class Seq2Seq(nn.Module):
    def __init__(self,config):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config).to(config.device)
        self.decoder = Decoder(config).to(config.device)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr)

        self.criterion = nn.MSELoss()

    def train(self,model,config,dataloader):
        print('==>开始训练...')

        logging_name = config.model_name + '_T:' + str(config.ntime_steps) + '_D:' + str(config.n_next)
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', filename='./log/' + logging_name + '.log',
                            level=logging.INFO)

        best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
        all_epoch = 0  # 记录进行了多少个epoch
        last_imporve = 0  # 记录上次校验集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升，停止训练

        start_time = time.time()

        for epoch in range(config.epochs):

            for i,train_data in enumerate(dataloader):

                train_x ,train_y= torch.as_tensor(train_data[0],dtype=torch.float32).to(config.device),torch.as_tensor(train_data[1],dtype=torch.float32).to(config.device)
                # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                train_y = train_y.squeeze(2)   #将最后的1去掉

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # Reset hidden state of encoder for current batch
                self.encoder.hidden = self.encoder.init_hidden(train_x.shape[0],config)

                # Do forward pass through encoder
                hidden = self.encoder(train_x)
                # Do forward pass through decoder (decoder gets hidden state from encoder)
                output , loss = self.decoder(train_y, hidden, self.criterion,config)
                # Backpropagation
                loss.backward()
                # Update parameters
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()


            # self.test(epoch,config,model)

            if all_epoch % 10 == 0:
                self.test_model(config)  # 可视化训练中的实际情况
                if loss < best_loss:
                    best_loss = loss
                    # torch.save(model.state_dict(), config.save_model)  #只保存模型的参数
                    torch.save(model,config.save_model)  #保存整个模型
                    imporve = "*"
                    last_imporve = all_epoch
                else:
                    imporve = " "
                time_dif = utils.get_time_dif(start_time)

                MSE, RMSE, MAE = evaluation(train_y.cpu(),output.detach().cpu().numpy())

                msg = 'Epochs:{0:d}, Loss:{1:.5f}, MSE:{2:.5f}, RMSE:{3:.5f}, MAE:{4:.5f}, Time:{5} {6}'

                print(msg.format(epoch,loss.item(), MSE,RMSE,MAE,time_dif, imporve))

                logging.info(msg.format(epoch,loss.item(), MSE,RMSE,MAE,time_dif, imporve))

            all_epoch = all_epoch + 1

            if all_epoch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

            if flag:
                break

    def test_model(self,config):
        # 加载全部数据进行测试
        # print("*" * 100)
        # print("==>加载已训练好的模型...")

        # model.load_state_dict(torch.load(config.save_model, map_location=torch.device(config.device)))

        # model = torch.load(config.save_model)

        y_train_pred, Y1 = self.draw_pic(config, on_train=True)
        y_test_pred, Y1 = self.draw_pic(config, on_train=False)

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


    def draw_pic(self,config,on_train=True):
        # encoder = Encoder(config).to(config.device)
        # decoder = Decoder(config).to(config.device)
        #
        # criterion = nn.MSELoss()

                # Train_X, Train_Y, Test_X, Test_Y = get_data(config.ntime_steps, config.n_next)

        X1,Y1 = read_data(config.dataroot)
        train_timesteps = int(X1.shape[0] * 0.8)

        if on_train:
            y_pred = np.zeros(train_timesteps - config.ntime_steps + 1)
        else:
            y_pred = np.zeros(X1.shape[0] - train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + config.batch_size)]
            X = np.zeros((len(batch_idx), config.ntime_steps - 1, X1.shape[1]))
            y_history = np.zeros((len(batch_idx), config.ntime_steps - 1))

            for j in range(len(batch_idx)):
                if on_train:

                    X[j, :, :] = X1[range(batch_idx[j], batch_idx[j] + config.ntime_steps - 1), :]

                    y_history[j, :] = Y1[range(batch_idx[j], batch_idx[j] + config.ntime_steps - 1)]

                else:
                    X[j, :, :] = X1[range(batch_idx[j] + train_timesteps - config.ntime_steps, batch_idx[j] + train_timesteps - 1), :]

                    y_history[j, :] = Y1[range(batch_idx[j] + train_timesteps - config.ntime_steps, batch_idx[j] + train_timesteps - 1)]

                    #################这里用不到#############
            y_history = torch.from_numpy(y_history).type(torch.FloatTensor).to(config.device)

                    # Reset hidden state of encoder for current batch
            self.encoder.hidden = self.encoder.init_hidden(X.shape[0], config)

                    # Do forward pass through encoder
            hidden = self.encoder(torch.tensor(X, dtype=torch.float32).to(config.device))
                    # Do forward pass through decoder (decoder gets hidden state from encoder)
                    # hidden[0] = hidden[0].permute(0,2,1)
                    # hidden[1] = hidden[1].permute(0,2,1)

            output, loss = self.decoder(y_history, hidden, self.criterion, config)
                    #########################
            # output = model(torch.tensor(X, dtype=torch.float32).to(config.device))
            y_pred[i:(i + config.batch_size)] = output.detach().cpu().numpy()[:,0]


                    # y_pred[i:(i+config.batch_size)] = model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()[:,0]

                    # _, input_encoded = self.Encoder((torch.from_numpy(X).type(torch.FloatTensor).to(config.device)))
                    #
                    # y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,y_history).cpu().data.numpy()[:, 0]
            i += config.batch_size

        return y_pred,Y1

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