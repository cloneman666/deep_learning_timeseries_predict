#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-09 16:24
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : train.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
# from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import utils


def train(model,config,train_dataloader,test_dataloader): #此处可以加入测试数据的参数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    print('==>开始训练...')
    best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
    all_epoch = 0  # 记录进行了多少个epoch
    last_imporve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练

    start_time = time.time()

    for epoch in range(config.epochs):

        for i, train_data in enumerate(train_dataloader):
            train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32), torch.as_tensor(train_data[1],
                                                                                                    dtype=torch.float32)
            # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
            train_y = train_y.squeeze(2)  # 将最后的1去掉

            optimizer.zero_grad()

            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            plt.ion()
            plt.figure()

            plt.plot(range(1, 1 + len(train_y)), train_y, label="True")
            plt.plot(range(1, 1 + len(output.detach().numpy())), output.detach().numpy(), label="Test")
            plt.legend()
            plt.pause(1)
            plt.close()

        if all_epoch % 50 == 0:
            with torch.no_grad():
                for i , test_data in enumerate(test_dataloader):
                    test_x, test_y = torch.as_tensor(test_data[0], dtype=torch.float32), torch.as_tensor(
                        test_data[1],
                        dtype=torch.float32)
                    # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                    test_y = test_y.squeeze(2)  # 将最后的1去掉
                    test_output = model(test_x)
                    test_loss = criterion(test_output,test_y)
                    test_MSE, test_RMSE, test_MAE = evaluation(test_y, test_output.detach().numpy())

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), config.save_model)
                imporve = "*"
                last_imporve = all_epoch
            else:
                imporve = " "
            time_dif = utils.get_time_dif(start_time)

            MSE, RMSE, MAE = evaluation(train_y, output.detach().numpy())

            msg = 'Epochs:{0:d}, Train Loss:{1:.5f},Test Loss:{2:.5f}, Train_MSE:{3:.5f}, Train_RMSE:{4:.5f}, Train_MAE:{5:.5f},Test_MSE:{6:.5f}, Test_RMSE:{7:.5f}, Test_MAE:{8:.5f}, Time:{9} {10}'

            print(msg.format(epoch, loss.item(),test_loss.item(), MSE, RMSE, MAE,test_MSE, test_RMSE, test_MAE, time_dif, imporve))


        all_epoch = all_epoch + 1

        if all_epoch - last_imporve > config.require_improvement:
            # 在验证集合上loss超过1000batch没有下降，结束训练
            print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
            flag = True
            break

        if flag:
            break


def test(model,config,train_data,test_data):
    pass

def evaluate(model,config,train_data,test_data):
    pass


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


# def testhhh(config,on_train=False):
#         """Prediction."""
#
#     if on_train:
#         y_pred = np.zeros(self.train_timesteps - self.T + 1)
#     else:
#         y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)
#
#     i = 0
#     while i < len(y_pred):
#         batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
#         X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
#         y_history = np.zeros((len(batch_idx), self.T - 1))
#
#         for j in range(len(batch_idx)):
#             if on_train:
#                 X[j, :, :] = self.X[range(
#                     batch_idx[j], batch_idx[j] + self.T - 1), :]
#                 y_history[j, :] = self.y[range(
#                     batch_idx[j], batch_idx[j] + self.T - 1)]
#             else:
#                 X[j, :, :] = self.X[range(
#                         batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
#                 y_history[j, :] = self.y[range(
#                         batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]
#
#         y_history = (torch.from_numpy(y_history).type(torch.FloatTensor).to(self.device))
#
#         _, input_encoded = self.Encoder((torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
#
#         y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,y_history).cpu().data.numpy()[:, 0]
#         i += self.batch_size
#
#     return y_pred