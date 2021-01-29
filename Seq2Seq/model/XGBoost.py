#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/14/20 11:01 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : XGBoost.py
# @Software: PyCharm


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from utils import *
import time
import logging
logging.basicConfig(filename='./log/XGBoost.log', level=logging.INFO)


def train_XGBoost(X_train,Y_train,X_test,Y_test,ALL_Y):
    print('========>XGBoost算法')
    model = XGBRegressor(n_estimators=1000)
    start_time = time.time()

    model.fit(X_train,Y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_MSE, train_RMSE, train_MAE = evaluation(y_pred_train, Y_train)

    test_MSE, test_RMSE, test_MAE = evaluation(y_pred_test, Y_test)

    time_dif = get_time_dif(start_time)

    msg = 'Train_MSE:{0:.5f}, Train_RMSE:{1:.5f}, Train_MAE:{2:.5f},Test_MSE:{3:.5f}, Test_RMSE:{4:.5f}, Test_MAE:{5:.5f},Time:{6}'
    print(msg.format(train_MSE, train_RMSE, train_MAE, test_MSE, test_RMSE, test_MAE, time_dif))


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
