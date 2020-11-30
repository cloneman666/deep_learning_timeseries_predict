#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 11/30/20 5:25 PM
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : ARIMA.py
# @Software: PyCharm

from statsmodels.tsa.arima_model import ARIMA


def train(X_train,Y_train,X_test,Y_test,ALL_Y):
    model = ARIMA(Y_train,order=(2,1,2))


