#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/29/20 8:24 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : Random_Forest.py
# @Software: PyCharm

#

# https://www.cnblogs.com/nolonely/p/7007961.html

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from utils import *
import matplotlib.pyplot as plt
import time
import logging
# 通过下面的方式进行简单配置输出方式与日志级别
logging.basicConfig(filename='./log/RF.log', level=logging.INFO)

#随机森林的训练方法
def train(X_train,Y_train):  # ,X_test,Y_test

    model = RandomForestRegressor()

    print("==========下面是RandomizedSearchCV的测试结果===============")
    logging.info("==========下面是RandomizedSearchCV的测试结果===============")
    # 设置想要优化的超参数以及他们的取值分布
    n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]

    param_dist={
        "n_estimators": n_estimators,
        "max_depth": [2,3,4, None],
        "max_features": sp_randint(1, 11),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "bootstrap": [True, False],
    }
    n_iter_search = 20
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search)

    start_time = time.time()  # 记录开始时间
    random_search.fit(X_train,Y_train)

    time_dif = get_time_dif(start_time)  # 计算时间间隔
    print(f'RandomizedSearchCV 一共迭代：{n_iter_search} 花费时间:{time_dif}')

    report(random_search.cv_results_)
    logging.info(report(random_search.cv_results_))

    print("==========下面是GridSearchCV的测试结果===============")
    logging.info("==========下面是GridSearchCV的测试结果===============")
    # 在所有参数上搜索，找遍所有网络节点
    param_grid = {
                  "n_estimators": n_estimators,
                  "max_depth": [2,3,4, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  }
    # 开启超参数空间的网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid)
    start_time2 = time.time()
    grid_search.fit(X_train,Y_train)

    time_dif2 = get_time_dif(start_time2)  # 计算时间间隔

    print(f'GridSearchCV 花费时间:{time_dif2} 参数设置:{len(grid_search.cv_results_)}')
    report(random_search.cv_results_)
    logging.info(report(random_search.cv_results_))
    # model.fit(X_train,Y_train)
    #
    # y_pred_train = model.predict(X_train)
    # # y_pred_test = model.predict(X_test)
    #
    # train_MSE, train_RMSE, train_MAE = evaluation(y_pred_train,Y_train)

    # test_MSE ,test_RMSE,test_MAE = evaluation(y_pred_test,Y_test)



    # msg = 'Train_MSE:{0:.5f}, Train_RMSE:{1:.5f}, Train_MAE:{2:.5f},Time:{3}'
    # print(msg.format(train_MSE, train_RMSE, train_MAE,time_dif))
    # logging.info(msg.format(train_MSE, train_RMSE, train_MAE,time_dif))




def report(results,n_top=3):
    for i in range(1,n_top+1):
        candidates=np.flatnonzero(results['rank_test_score']==i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters:{0}".format(results['params'][candidate]))
            print("")


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