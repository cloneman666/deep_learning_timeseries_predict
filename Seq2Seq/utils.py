#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-07 22:09
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : utils.py
# @Software: PyCharm

import time
from datetime import timedelta


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))