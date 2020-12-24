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
from torch.autograd import Variable
from model.utils import *
import time
import matplotlib.pyplot as plt
import utils
import logging

from tensorboardX import SummaryWriter


def train(model,config,train_dataloader,test_dataloader): #此处可以加入测试数据的参数
    writer = SummaryWriter('./log/'+ config.model_name + '_T:' +str(config.ntime_steps) + '_D:'+ str(config.n_next))

    logging_name = config.model_name + '_T:'+str(config.ntime_steps) +'_D:'+ str(config.n_next)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', filename='./log/'+logging_name+'.log', level=logging.INFO)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    print('==>开始训练...')

    logging.info('使用模型：'+config.model_name)
    logging.info("==> Use accelerator: "+str(config.device))

    best_loss = float('inf')  # 记录最小的损失，，这里不好加一边训练一边保存的代码，无穷大量
    all_epoch = 0  # 记录进行了多少个epoch
    last_imporve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练

    start_time = time.time()
    model.train()
    for epoch in range(config.epochs):

        for i, train_data in enumerate(train_dataloader):
            train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32).to(
                        config.device), torch.as_tensor(train_data[1], dtype=torch.float32).to(config.device)
                    # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
            train_y = train_y.squeeze(2)  # 将最后的1去掉

            optimizer.zero_grad()

            if config.model_name == 'Seq2Seq+Att':  # 训练的时候用于选择模型
                output = model(train_x, train_y)
            else:
                output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()

                # 一边训练一边可视化
                # if epoch % 50 == 0:
                #     y_train_pred ,Y1= draw_pic(model,config,on_train=True)
                #     y_test_pred ,Y1= draw_pic(model,config,on_train=False)
                #
                #     plt.ion()
                #     plt.figure()
                #     plt.plot(range(1, 1 + len(Y1)), Y1, label='True')
                #     plt.plot(range(config.ntime_steps + len(y_train_pred), len(Y1) + 1),y_test_pred, label='Predicted - Test')
                #
                #     plt.plot(range(config.ntime_steps, len(y_train_pred) + config.ntime_steps),y_train_pred, label='Predicted - Train')
                #
                #     plt.legend()
                #     plt.pause(1)
                #     plt.close()

        if all_epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                for i, test_data in enumerate(test_dataloader):
                    test_x, test_y = torch.as_tensor(test_data[0], dtype=torch.float32).to(
                                        config.device), torch.as_tensor(
                                        test_data[1],
                                        dtype=torch.float32).to(config.device)
                                    # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                    test_y = test_y.squeeze(2)  # 将最后的1去掉

                    if config.model_name == 'Seq2Seq+Att':
                        test_output = model(test_x, test_y)
                    else:
                        test_output = model(test_x)

                    test_loss = criterion(test_output, test_y)
                    test_MSE, test_RMSE, test_MAE = evaluation(test_y.cpu().numpy(),
                                                                       test_output.detach().cpu().numpy())

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), config.save_model)
                imporve = "*"
                last_imporve = all_epoch
            else:
                imporve = " "
            time_dif = utils.get_time_dif(start_time)

            MSE, RMSE, MAE = evaluation(train_y.cpu().numpy(), output.detach().cpu().numpy())

            msg = 'Epochs:{0:d}, Train Loss:{1:.5f},Test Loss:{2:.5f}, Train_MSE:{3:.5f}, Train_RMSE:{4:.5f}, Train_MAE:{5:.5f},Test_MSE:{6:.5f}, Test_RMSE:{7:.5f}, Test_MAE:{8:.5f}, Time:{9} {10}'

            print(msg.format(epoch, loss.item(),test_loss.item(), MSE, RMSE, MAE,test_MSE, test_RMSE, test_MAE, time_dif, imporve))

            writer.add_scalar('train/loss',loss.item(),epoch)
            writer.add_scalars('train/scalars',{'MSE':MSE,'RMSE':RMSE,'MAE':MAE},epoch)
            # writer.add_scalar('train/RMSE',RMSE,epoch)
            # writer.add_scalar('train/MAE',MAE,epoch)

            #记录到日志文件
            logging.info(msg.format(epoch, loss.item(),test_loss.item(), MSE, RMSE, MAE,test_MSE, test_RMSE, test_MAE, time_dif, imporve))

        all_epoch = all_epoch + 1

        if all_epoch - last_imporve > config.require_improvement:
                    # 在验证集合上loss超过1000batch没有下降，结束训练
            print('==>在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
            writer.close()
            flag = True
            break

        if flag:
            writer.close()
            break

def draw_pic(model,config,on_train=True):
    """
    此函数为画图函数，将每步的图像进行可视化
    """


    # Train_X, Train_Y, Test_X, Test_Y = get_data(config.ntime_steps, config.n_next)
    pic_data = read_all_data(config.dataroot)

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

        #这里用不到
        # y_history = torch.from_numpy(y_history).type(torch.FloatTensor).to(config.device)

        # train_x, train_y = torch.as_tensor(train_data[0], dtype=torch.float32), torch.as_tensor(train_data[1],
        #                                                                                         dtype=torch.float32)
        # train_x = train_x.transpose(1,0)  # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
        # train_y = train_y.squeeze(2)  # 将最后的1去掉

        if config.model_name=='Seq2Seq+Att':
            y_pred[i:(i+config.batch_size)] = model(torch.as_tensor(X, dtype=torch.float32).to(config.device),torch.as_tensor(y_history,dtype=torch.float32).to(config.device)).detach().cpu().numpy()[:, 0]
        else:
            y_pred[i:(i+config.batch_size)] = model(torch.as_tensor(X, dtype=torch.float32).to(config.device)).detach().cpu().numpy()[:,0]



        # _, input_encoded = self.Encoder((torch.from_numpy(X).type(torch.FloatTensor).to(config.device)))
        #
        # y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,y_history).cpu().data.numpy()[:, 0]
        i += config.batch_size

    return y_pred,Y1,pic_data

def test(model,config):
    print("*"*100)
    print("==>加载已训练好的模型...")

    model.load_state_dict(torch.load(config.save_model,map_location=torch.device(config.device)))

    y_train_pred, Y1,pic_data = draw_pic(model, config, on_train=True)
    y_test_pred, Y1,_ = draw_pic(model, config, on_train=False)

    # plt.ion()
    fig,(ax0,ax1) = plt.subplots(nrows=2)
    ax0.set_title(config.model_name+'_T:' + str(config.ntime_steps)+ '_D:'+str(config.n_next))  #
    # ax0.plot(range(1, 1 + len(Y1)), Y1, label='Ground Truth')
    ax0.plot(pic_data.ds,pic_data.y, label='Ground Truth')
    ax0.plot(pic_data.ds[config.ntime_steps:config.ntime_steps +len(y_train_pred)],y_train_pred,alpha=0.5,label='Predicted - Train')
    ax0.plot(pic_data.ds[config.ntime_steps+ len(y_train_pred)-1:],y_test_pred,alpha=0.5,label='Predicted - Test')

    ax1.set_title('Magnifies test set predictions')
    ax1.plot(pic_data.ds[config.ntime_steps+ len(y_train_pred)-1:],pic_data.y[config.ntime_steps+ len(y_train_pred)-1:], label='Ground Truth')
    ax1.plot(pic_data.ds[config.ntime_steps+ len(y_train_pred)-1:],y_test_pred,'g',alpha=0.5,label='Predicted - Test')

    for tick in ax1.get_xticklabels():
        tick.set_rotation(25)
    # ax0.plot(range(config.ntime_steps + len(y_train_pred), len(Y1) + 1), y_test_pred,alpha=0.5,label='Predicted - Test')
    # ax0.plot(range(config.ntime_steps, len(y_train_pred) + config.ntime_steps), y_train_pred,alpha=0.4,label='Predicted - Train')
    #

    # ax1.plot(range(1, len(y_test_pred)),Y1[len(y_train_pred)+config.ntime_steps:],label='Ground Truth')
    #
    # ax1.plot(range(1 ,1 + len(y_test_pred)),y_test_pred,label='Predicted - Test')

    # plt.figure(figsize=(10,3),dpi=300)
    # plt.title(config.model_name+'_T:'+str(config.ntime_steps) +'_D:'+str(config.n_next))
    # plt.plot(range(1, 1 + len(Y1)), Y1, label='Ground Truth')
    # plt.plot(range(config.ntime_steps + len(y_train_pred), len(Y1) + 1), y_test_pred, label='Predicted - Test')
    #
    # plt.plot(range(config.ntime_steps, len(y_train_pred) + config.ntime_steps), y_train_pred, label='Predicted - Train')
    # plt.tight_layout()
    # plt.legend()
    # # plt.pause(1)
    # # plt.close()
    # plt.savefig('./data/pic/'+config.model_name +'_T:'+str(config.ntime_steps) +'_D:'+str(config.n_next)+'.png')
    ax0.legend()
    ax1.legend()
    plt.tight_layout()
    plt.savefig('./data/pic/'+config.model_name +'_T:'+str(config.ntime_steps) +'_D:'+str(config.n_next)+'.png',dpi=300)  #
    plt.show()

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


def draw_model_structure(model,config):
    print("*" * 100)

    model.load_state_dict(torch.load(config.save_model, map_location=torch.device(config.device)))

    print("===>已经加载训练好的模型结构："+config.model_name)
    dummy_input = torch.rand(128, 10, 20)

    with SummaryWriter(comment='Net') as w:
        w.add_graph(model,(dummy_input, ))
    print("===>已经生成可视化的模型结构！")



