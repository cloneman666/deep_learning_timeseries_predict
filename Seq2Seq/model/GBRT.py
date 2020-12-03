#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 12/2/20 4:25 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : GBRT.py
# @Software: PyCharm

import numpy as np
import time
from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from utils import *
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='./log/GBRT.log', level=logging.INFO)

def GetBestWay(model,X_train, Y_train):
    print("==========下面是RandomizedSearchCV的测试结果===============")
    logging.info("==========下面是RandomizedSearchCV的测试结果===============")
    # 设置想要优化的超参数以及他们的取值分布
    n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]

    param_dist = {
        "n_estimators": n_estimators,
        "max_depth": [2,3,4,5, None],
        "max_features": range(1, 11,1),
        "min_samples_split": range(2, 11,1),
        "min_samples_leaf": range(1, 11,1),
        "learning_rate":[0.01,0.1,0.001]
    }
    n_iter_search = 100
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search)

    start_time = time.time()  # 记录开始时间
    random_search.fit(X_train, Y_train)

    time_dif = get_time_dif(start_time)  # 计算时间间隔
    print(f'RandomizedSearchCV 一共迭代：{n_iter_search} 花费时间:{time_dif}')

    report(random_search.cv_results_)

    print("==========下面是GridSearchCV的测试结果===============")
    logging.info("==========下面是GridSearchCV的测试结果===============")
    # 在所有参数上搜索，找遍所有网络节点
    param_grid = {
        "max_depth": [2,3,4,5, None],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "learning_rate": [0.01, 0.1, 0.001]
    }
    # 开启超参数空间的网格搜索
    grid_search = GridSearchCV(model, param_grid=param_grid)
    start_time2 = time.time()
    grid_search.fit(X_train, Y_train)

    time_dif2 = get_time_dif(start_time2)  # 计算时间间隔

    print(f'GridSearchCV 花费时间:{time_dif2} 参数设置:{len(grid_search.cv_results_)}')
    report(random_search.cv_results_)


def train_GBRT(X_train,Y_train,X_test,Y_test,ALL_Y):
    print('=======>GBRT算法')
    model = GradientBoostingRegressor(
        n_estimators=250,
        min_samples_split=6,
        min_samples_leaf=1,
        max_features=9,
        max_depth=2,
        learning_rate=0.01
    )

    # n_estimators
    # ': 250, '
    # min_samples_split
    # ': 6, '
    # min_samples_leaf
    # ': 1, '
    # max_features
    # ': 9, '
    # max_depth
    # ': 2, '
    # learning_rate
    # ': 0.01}

    start_time = time.time()

    # GetBestWay(model, X_train, Y_train)

    model.fit(X_train,Y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f'训练的精确度：{model.score(X_train,Y_train)}')
    print(f'测试的精确度：{model.score(X_test,Y_test)}')

    train_MSE, train_RMSE, train_MAE = evaluation(y_pred_train, Y_train)

    test_MSE, test_RMSE, test_MAE = evaluation(y_pred_test, Y_test)

    time_dif = get_time_dif(start_time)

    msg = 'Train_MSE:{0:.5f}, Train_RMSE:{1:.5f}, Train_MAE:{2:.5f},Test_MSE:{3:.5f}, Test_RMSE:{4:.5f}, Test_MAE:{5:.5f},Time:{6}'
    print(msg.format(train_MSE, train_RMSE, train_MAE, test_MSE, test_RMSE, test_MAE, time_dif))

    plt.figure(figsize=(10, 3), dpi=300)
    plt.title('GBRT Predicts Ticket Prices')

    plt.plot(range(len(ALL_Y)), ALL_Y, label='Ground Truth')

    plt.plot(range(len(Y_train)), y_pred_train, label='Predicted - Train')
    plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), y_pred_test, label='Predicted - Test')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./data/pic/GBRT.png')
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
