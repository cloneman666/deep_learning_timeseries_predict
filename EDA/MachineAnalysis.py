#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-10-27 08:33
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : MachineAnalysis.py
# @Software: PyCharm

from sklearn import linear_model


class MachineAnalysis():
    def __init__(self,dataframe):
        self.df = dataframe
        self.columns = dataframe.columns
        self.numerical_columns = [name for name in self.columns if
                                  (self.df[name].dtype == 'int64') | (self.df[name].dtype == 'float64')]

    def Ordinary_Least_Squares(self):
        pass



