#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-02 17:12
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : test.py
# @Software: PyCharm


import pymysql
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 从远程获取数据
def get_data1():
    try:
        #co = pymysql.connect(host="192.168.2.51", user="ggy", db="ky", passwd="123456", use_unicode=True, charset="utf8")
        sql = "select * from fpnr limit 100"
        print("正在加载数据，请稍后！")
        con = pymysql.connect(host='localhost',port = 3306,user='root',db="airportdata",passwd='zzc123456',use_unicode=True,charset="utf8")
        # 获取游标
    # cursor = con.cursor()
    # cursor.execute(sql)
        data = pd.read_sql(sql,con)

    except:
        print("数据加载异常！")

    # except:
    #     print("数据加载异常！")
    return data

def get_data2():
    try:
        sql = 'select * from prl limit 100'
        print('数据加载中....')
        con = pymysql.connect(host='localhost', port=3306, user='root', db="airportdata", passwd='zzc123456',
                          use_unicode=True, charset="utf8")
        data = pd.read_sql(sql,con)
    except:
        print("数据加载异常！")

    return data

# 数据预处理表1，fpnr
def process_data(datafile):
    #这一部分为其中一个表的预处理
    # df1 = pd.read_csv(datafile, usecols=["Pnr_Crt_Dt", "Lcl_Dpt_Dt", "Lcl_Dpt_Tm", "Pax_Id_Nbr"])
    df1 = pd.DataFrame(datafile,columns=["Pnr_Crt_Dt", "Lcl_Dpt_Dt", "Lcl_Dpt_Tm", "Pax_Id_Nbr"])
    data = df1.drop_duplicates()
    data1 = data.dropna(axis=0, how='any')

    data_deal = data1.drop(['Lcl_Dpt_Tm'], axis=1)
    data_deal["Pnr_Crt_Dt"] = pd.to_datetime(data_deal["Pnr_Crt_Dt"], format="%Y%m%d")
    data_deal["Lcl_Dpt_Dt"] = pd.to_datetime(data_deal["Lcl_Dpt_Dt"], format="%Y%m%d")
    data_deal["提前购票天数"] = data_deal["Lcl_Dpt_Dt"] - data_deal["Pnr_Crt_Dt"]

    data_deal['提前购票天数'] = data_deal['提前购票天数'].astype(np.int64) / (60 * 60 * 24 * 10 ** 9)
    data_deal.rename(columns={'提前购票天数': 'Pre_buy_date'}, inplace=True)

    data_mean = data_deal.groupby("Pax_Id_Nbr").Pre_buy_date.mean()
    data_mean = data_mean.reset_index()  # 重新设置index，将原来的index作为counts的一列。
    # data11.columns = ['index', 'num'] #重新设置列名，主要是第二列，默认为0
    data_mean.columns = ['identityNum', 'avg_day']

    return data_mean

#数据预处理表2,prl
def process_data2(datafile):
    # df = pd.read_csv('prl.csv', error_bad_lines=False)

    df = pd.DataFrame(datafile)
    data_deal = df.drop(
        ['recordId', 'firstName', 'lastName', 'name', 'cnin', 'tkne', 'legNum', 'psm', 'infTkne1', 'legNum1',
         'infName1', 'infBirth1', 'psm1', 'infTkne2', 'legNum2', 'infName2', 'infBirth2', 'psm2', 'hc', 'vip', 'm', 'n',
         'o', 'ckin', 'pspt', 'doca', 'asvc', 'rn'], axis=1)

    data = data_deal.drop_duplicates()
    data1 = data.drop(
        ['identity', 'psptDate', 'fqtvMark', 'flightNo', 'depAirpt', 'arrvAirpt', 'seatRecord', 'pasType', 'pasSex',
         'seat', 'bn', 'gate', 'psptNum', 'psptOrigo', 'psptOrigo', 'fqtv', 'bugSum', 'bugWegit', 'docs'], axis=1)

    counts_identityNum = data1['identityNum'].value_counts()

    counts_identityNum = counts_identityNum.reset_index()  # 重新设置index，将原来的index作为counts的一列。
    counts_identityNum.columns = ['identityNum', 'num']  # 重新设置列名，主要是第二列，默认为0
    counts_identityNum.groupby('identityNum')

    counts_identityNum_sort = counts_identityNum.sort_values(by="identityNum", ascending=False)

    data1['order_day'] = pd.to_datetime(data1.flightDate, format="%Y%m%d")
    data1['day'] = data1.order_day.values.astype('datetime64[D]')

    data_R = data1.groupby("identityNum").day.max()

    data_R = data_R.reset_index()  # 重新设置index，将原来的index作为counts的一列。
    # data11.columns = ['index', 'num'] #重新设置列名，主要是第二列，默认为0
    data_R.columns = ['identityNum', 'last_data']

    end_time = 20181001
    data_R['end_day'] = pd.to_datetime(end_time, format="%Y%m%d")

    data_R["end_day"] = pd.to_datetime(data_R["end_day"])
    data_R["last_data"] = pd.to_datetime(data_R["last_data"])
    data_R["R"] = data_R["end_day"] - data_R["last_data"]

    data_R['R'] = data_R['R'].astype(np.int64) / (60 * 60 * 24 * 10 ** 9)

    R = data_R.drop(['last_data', 'end_day'], axis=1)
    R_sort = R.sort_values(by="identityNum", ascending=False)

    data11 = data1.groupby("identityNum").day.max()

    data11 = data11.reset_index()  # 重新设置index，将原来的index作为counts的一列。
    # data11.columns = ['index', 'num'] #重新设置列名，主要是第二列，默认为0
    data11.columns = ['identityNum', 'last_data']

    end_time = 20181001
    data11['end_day'] = pd.to_datetime(end_time, format="%Y%m%d")

    data11["R"] = data11["end_day"] - data11["last_data"]

    # counts_R = data11.drop(['last_data', 'end_day', 'R_day', 'R1'], axis=1)
    # counts_R_sort = counts_R.sort_values(by="identityNum", ascending=False)
    #
    # counts_FR = pd.concat([counts_R_sort, counts_identityNum_sort])

    FR = pd.concat([R_sort, counts_identityNum_sort])
    FR = FR.fillna(0)

    FR_group = FR.groupby('identityNum').sum()

    FR_ok = FR_group.reset_index()  # 重新设置index，将原来的index作为counts的一列。
    FR_ok.columns = ['identityNum', 'R', 'F']  # 重新设置列名，主要是第二列，默认为0


    return FR_ok

def cat_data1_data2():
    pass



def main():
    pass



if __name__ == '__main__':
    data1 = get_data1()
    data_mean = process_data(data1)
    print(data_mean)

    data2 = get_data2()
    datahh = process_data2(data2)
    print(datahh)