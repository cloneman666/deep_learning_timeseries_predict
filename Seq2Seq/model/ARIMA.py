#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/30/20 5:25 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : ARIMA.py
# @Software: PyCharm

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings

import pmdarima as pm  #自动选择参数
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def train_ARIMA(X_train,Y_train,X_test,Y_test,ALL_Y):
    print('=======>移动平均方法')
    # model = ARIMA(Y_train,order=(2,0,1))
    # model_fit = model.fit(disp=0)

    # 自动选择参数
    model = pm.auto_arima(Y_train, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,    # No Seasonality
                      start_P=0,
                      D=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

    print(model.summary())

    #存储往后预测的天数的动态图
    # ARIMA_list = []
    # Forecast
    # for i in range(1,199):

    n_periods = 198   #这里最大填：198  数据的关系
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)

    MSE, RMSE, MAE = evaluation(Y_test[:n_periods], fc)
    msg = 'MSE:{0:.5f}, RMSE:{1:.5f}, MAE:{2:.5f}'

    print(msg.format(MSE, RMSE, MAE))
    # ARIMA_list.append(msg.format(MSE, RMSE, MAE))

    # df =pd.DataFrame(ARIMA_list)
    # df.to_csv('./data/ARIMA.csv')

    #后面为画图必要的处理措施
    index_of_fc = np.arange(len(Y_train), len(Y_train) + n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(10, 3), dpi=300)
    plt.title('ARIMA')
    # plt.plot()

    plt.plot(ALL_Y, label='Ground Truth')

    plt.plot(fc_series, color='darkgreen', label='ARIMA Predicts Price Trends')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    # plt.title("Final Forecast of WWW Usage")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./data/pic/ARIMA.jpg')
    # # plt.close()
    plt.show()

def draw_arima():
    data = pd.read_csv('./data/ARIMA.csv').drop('Unnamed: 0',axis=1)
    hh = data['0'].str.split(',',expand=True) #.str.split(':')
    hh.columns = ['n_periods','MSE','RMSE','MAE']
    # data2 = hh.rename(columns={'0':'n_periods','1':'MSE','2':'RMSE','3':'MAE'})
    # data['MSE'] = data['0'].str.slice(17,24)

    hh['n_periods'] = hh['n_periods'].str.slice(10,14).astype('float')
    hh['MSE'] = hh['MSE'].str.slice(4,15).astype('float')
    hh['RMSE'] = hh['RMSE'].str.slice(6,15).astype('float')
    hh['MAE'] = hh['MAE'].str.slice(5,15).astype('float')
    hh.drop('n_periods',axis=1)

    print(hh.describe())

    # print(hh.info())
    # plt.figure(figsize=(10,8),dpi=300)
    plt.title('ARIMA')
    plt.plot(hh.index,hh['MSE'],label='MSE')
    plt.plot(hh.index,hh['RMSE'],label='RMSE')
    plt.plot(hh.index,hh['MAE'],label='MAE')
    plt.xlabel('Future time step')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/pic/ARIMA_test.png',dpi=300)
    plt.show()
    # for i in range():
    #
    #
    #
    # print(data.info())


    # MSE, RMSE, MAE = evaluation(Y_train,y_prd_train)
    # test_MSE, test_RMSE, test_MAE = evaluation(Y_test,y_prd_test)
    # msg = 'Train_MSE:{0:.5f}, Train_RMSE:{1:.5f}, Train_MAE:{2:.5f},Test_MSE:{3:.5f}, Test_RMSE:{4:.5f}, Test_MAE:{5:.5f}, Time:{6}'
    #
    # print(msg.format(MSE, RMSE, MAE,test_MSE, test_RMSE, test_MAE))





    # evaluate an ARIMA model for a given order (p,d,q)



#     def evaluate_arima_model(X, arima_order):
#         # prepare training dataset
#         train_size = int(len(X) * 0.8)
#         train, test = X[0:train_size], X[train_size:]
#         history = [x for x in train]
#         # make predictions
#         predictions = list()
#         for t in range(len(test)):
#             model = ARIMA(history, order=arima_order)
#             model_fit = model.fit(disp=0)
#             yhat = model_fit.forecast()[0]
#             predictions.append(yhat)
#             history.append(test[t])
#         # calculate out of sample error
#         error = mean_squared_error(test, predictions)
#         return error
#
#     # evaluate combinations of p, d and q values for an ARIMA model
#     def evaluate_models(dataset, p_values, d_values, q_values):
#         dataset = dataset.astype('float32')
#         best_score, best_cfg = float("inf"), None
#         for p in p_values:
#             for d in d_values:
#                 for q in q_values:
#                     order = (p, d, q)
#                     try:
#                         mse = evaluate_arima_model(dataset, order)
#                         if mse < best_score:
#                             best_score, best_cfg = mse, order
#                         print('ARIMA%s MSE=%.3f' % (order, mse))
#                     except:
#                         continue
#         print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
#
# # evaluate parameters
# #     p_values = [0, 1, 2, 4, 6, 8, 10]
#     p_values = [3,5,7]
#     d_values = range(0, 3)
#     q_values = range(0, 3)
#     warnings.filterwarnings("ignore")
#     evaluate_models(ALL_Y, p_values, d_values, q_values)

