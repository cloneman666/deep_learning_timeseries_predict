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
from sklearn.model_selection import train_test_split

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
    # X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].values #.as_matrix()

    X = df.loc[:, [x for x in df.columns.tolist() if x != 'ds' and x != 'y' and x != 'start_time' and x != 'Cls_Cd_Y']].values

    y = np.array(df.y)

    return X, y


#自定义方法，除去部分内容
def read_all_data(input_path):
    df = pd.read_csv(input_path)
    # df = df.drop(['ds','Cls_Cd_Y','start_time'],axis=1)
    # print(df.info())
    df['ds'] = pd.to_datetime(df['ds'])
    return df



def get_data(ntime_steps,n_next,hasTest = True):
    """
    将数据划分为X 和 Y
    :param ntime_steps: 时间窗口
    :param n_next:    想要预测后面的天数
    :param hasTest :   为是否进行测试集的划分，默认为True
    :return:
    """
    import pandas as pd

    one_hot_data = pd.read_csv('./data/one_hot_甘.csv')

    ntime_steps = ntime_steps  # 时间窗口T

    n_next = n_next

    if hasTest:

        one_hot_data2 = one_hot_data.drop(['ds', 'Cls_Cd_Y', 'start_time'], axis=1)
        # nrows = int(one_hot_data2.shape[0] * 0.8)
        # one_hot_data2 = one_hot_data2[:nrows]

        y_data = one_hot_data2.iloc[:, -1:].values

        train_X,test_X,train_y,text_y = train_test_split(one_hot_data2,y_data,test_size=0.2,shuffle=False)

        # print(type(train_X))

        Train_x, _ = create_dataset(train_X.values, ntime_steps, n_next)

        _, Train_y = create_dataset(train_y, ntime_steps, n_next)

        Test_x, _ = create_dataset(test_X.values, ntime_steps, n_next)

        _, Test_y = create_dataset(text_y, ntime_steps, n_next)



        return Train_x,Train_y,Test_x,Test_y

    else: #后面为不需要进行分测试集
        one_hot_data2 = one_hot_data.drop(['ds', 'Cls_Cd_Y', 'start_time'], axis=1)

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
        a = data[i:(i + ntime_steps),:-1]  # 自定义X
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
    def __init__(self,data_X,data_Y):

        self.X = data_X
        self.Y = data_Y
        # self.hasTest = hasTest
        # if hasTest:
        #     self.Train_X,self.Train_Y,self.Test_X,self.Test_Y = get_data(ntime_steps,n_next,hasTest)
        # else:
        #     self.X ,self.Y = get_data(ntime_steps,n_next,hasTest)

    def __getitem__(self, item):

        return self.X[item],self.Y[item]

    def __len__(self):
        return len(self.X)