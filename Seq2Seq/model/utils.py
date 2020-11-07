#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 11:25
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : utils.py
# @Software: PyCharm


import numpy as np
import pandas as pd

def read_data(input_path, debug=False):
    """
    读取数据

    Args:
        input_path (str): 输入的数据为多维，其中有一列为最终的预测值

    Returns:
        X (np.ndarray): 各种特征
        y (np.ndarray): 最后输出的值

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    # X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].as_matrix()
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].values

    y = np.array(df.NDX)

    return X, y