#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 11:25
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : utils.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

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
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'ds' and x != 'y' and x != 'start_time']].values

    y = np.array(df.y)

    return X, y


#自定义方法，除去部分内容
def read_all_data(input_path):
    df = pd.read_csv(input_path)
    df = df.drop(['ds','Cls_Cd_Y','start_time'],axis=1)
    return df



def get_data(ntime_steps,n_next):
    """
    将数据划分为X 和 Y
    :param ntime_steps: 时间窗口
    :param n_next:    想要预测后面的天数
    :return:
    """
    import pandas as pd

    one_hot_data = pd.read_csv('./data/one_hot_甘.csv')
    one_hot_data2 = one_hot_data.drop(['ds', 'Cls_Cd_Y', 'start_time'], axis=1)

    ntime_steps = ntime_steps  # 时间窗口T

    n_next = n_next

    y_data = one_hot_data2.iloc[:, -1:].values

    Train_x, _ = create_dataset(one_hot_data2.drop('y', axis=1).values, ntime_steps,n_next)

    _, Train_y = create_dataset(y_data, ntime_steps,n_next)

    return Train_x,Train_y


# 划分数据集函数
def create_dataset(data, ntime_steps, n_next):
    """
    划分数据集，用前多少天，预测后面多少天
    :param data:
    :param ntime_steps:
    :param n_next:
    :return:
    """
    #     dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0] - ntime_steps - n_next - 1):
        a = data[i:(i + ntime_steps), :-1]  # 自定义X
        train_X.append(a)
        tempb = data[(i + ntime_steps):(i + ntime_steps + n_next)]

        #         tempb = data[(i+ntime_steps):(i+ntime_steps+n_next),-1:]  #自定义Y
        #         tempb = data[i:(i+ntime_steps),-1:]   #自定义Y
        train_Y.append(tempb)

    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')
    return train_X, train_Y


# 创建一个自己的数据集
class MyDataset(Dataset):
    """
    将数据集进行内部函数的划分
    """
    def __init__(self,ntime_steps,n_next):
        self.X ,self.Y = get_data(ntime_steps,n_next)

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __len__(self):
        return len(self.X)