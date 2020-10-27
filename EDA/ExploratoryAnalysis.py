#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-10-27 08:35
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : ExploratoryAnalysis.py
# @Software: PyCharm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import streamlit as st

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

# Exploratory Analysis Class
class ExploratoryAnalysis:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = dataframe.columns
        self.numerical_columns = [name for name in self.columns if
                                  (self.df[name].dtype == 'int64') | (self.df[name].dtype == 'float64')]

    def info(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def info2(self, column_target):
        df = self.df[column_target].value_counts().to_frame().reset_index()
        df.sort_values(by='index', inplace=True, ignore_index=True)
        df.rename(columns={'index': column_target, '{}'.format(column_target): "Values Frequency"}, inplace=True)
        return df

    def CountPlot(self, column_target, hue=None):
        sns.set(style="darkgrid",font='SimHei')
        return sns.countplot(x=column_target, data=self.df, hue=hue, palette='pastel')

    def HeatMapCorr(self):
        sns.set(style="darkgrid",font='SimHei')
        sns.set(font_scale=0.6)
        corr = self.df.corr()
        return sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)

    def DistPlot(self, column_target):
        sns.set(style="darkgrid",font='SimHei')
        return sns.distplot(self.df[column_target], color='c')

    def PairPlot(self, hue=None):
        sns.set(style="darkgrid",font='SimHei')
        return sns.pairplot(self.df, hue=hue, palette="coolwarm")

    def BoxPlot(self, column_x=None, column_y=None, hue=None):
        sns.set(style="darkgrid",font='SimHei')
        return sns.boxplot(x=column_x, y=column_y, hue=hue, data=self.df, palette="Set3")
