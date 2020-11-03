#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-11-02 17:12
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : test.py
# @Software: PyCharm

# Windows 安装modin
#https://www.jianshu.com/p/ac339a8ea0d0

# import os
#
# os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask


import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,filename='./log/test.log')

def get_time_dif(start_time):
    """
    获取使用时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离的平方，并返回
    """
    return np.sum(np.power(vecA - vecB, 2))


def test_Kmeans_nclusters(data_train):
    """
    计算不同的k值时，SSE的大小变化
    """
    data_train = data_train.values
    nums = range(2, 8)
    SSE = []
    for num in nums:
        sse = 0
        kmodel = KMeans(n_clusters=num, n_jobs=4)
        kmodel.fit(data_train)
        # 簇中心
        cluster_ceter_list = kmodel.cluster_centers_
        # 个样本属于的簇序号列表
        cluster_list = kmodel.labels_.tolist()
        for index in tqdm(range(len(data_train))):
            cluster_num = cluster_list[index]
            sse += distEclud(data_train[index, :], cluster_ceter_list[cluster_num])
        print("簇数是", num, "时； SSE是", sse)
        SSE.append(sse)
    return nums, SSE

# 从远程获取数据fpnr
def get_data1():
    try:
        #co = pymysql.connect(host="192.168.2.51", user="ggy", db="ky", passwd="123456", use_unicode=True, charset="utf8")
        sql = "select * from fpnr limit 10000"
        print("fpnr正在远程加载数据，请稍后！")
        con = pymysql.connect(host='localhost',port = 3306,user='root',db="airportdata",passwd='zzc123456',use_unicode=True,charset="utf8")
    # cursor = con.cursor()
    # cursor.execute(sql)
        data = pd.read_sql(sql,con)
        print("数据加载成功！")

    except Exception as e:
        print("数据加载异常！")
        print(e)
    finally:
        con.close()
    return data

#从远程获取数据prl
def get_data2():
    try:
        sql = 'select * from prl '
        print('prl数据远程加载中....')
        con = pymysql.connect(host='localhost', port=3306, user='root', db="airportdata", passwd='zzc123456',
                          use_unicode=True, charset="utf8")
        data = pd.read_sql(sql,con)
        print("数据加载成功！")

    except Exception as e:
        print("数据加载异常！")
        print(e)
    finally:
        con.close()

    return data

# 数据预处理表1，fpnr
def process_data(datafile):
    #这一部分为其中一个表的预处理
    print('fpnr数据预处理中...')
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
    data_mean.to_csv('./output_data/data_avgday.csv')
    print('fpnr数据处理成功!')
    return data_mean

#数据预处理表2,prl
def process_data2(datafile):
    # df = pd.read_csv('prl.csv', error_bad_lines=False)
    print('prl数据预处理中...')
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
    FR_ok.to_csv('./output_data/data_RF.csv')

    return FR_ok

#数据fpnr和prl结合
def cat_data1_data2(data_RF,data_avgday):
    print("fpnr和prl两表结合中...")
    RFAvg_day = pd.concat([data_RF, data_avgday])
    RFAvg_day = RFAvg_day.fillna(0)
    RFAvg_day_group = RFAvg_day.groupby('identityNum').sum()
    # RFAvg_day_group1 = RFAvg_day_group

    index1 = RFAvg_day_group['F'] != 0
    index2 = RFAvg_day_group['R'] != 0
    RFAvg_day_group2 = RFAvg_day_group[index1 & index2]
    RFAvg_day_group2.to_csv('./output_data/RFAvg_day.csv')
    print('两表结合完成！')
    return RFAvg_day_group2



def main(data):
    print('开始聚类分析...')
    # data_RFAvg_day = pd.read_csv("RFAvg_day.csv", usecols=['identityNum', 'R', 'F', 'avg_day'])
    data_RFAvg_day = pd.DataFrame(data, columns=['identityNum', 'R', 'F', 'avg_day'])

    R = data_RFAvg_day.drop(['identityNum'], axis=1)

    data_zs = 1.0 * (R - R.mean()) / R.std()  # 数据标准化

    print(data_zs)

    k=3
    model = KMeans(n_clusters=k, n_jobs=4)  # 分为k类，并发数4
    model.fit(data_zs)  # 开始聚类

    # 简单打印结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    # print("model.labels:")
    # print(model.labels_)
    print("r1:")
    print(r1)
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    print("r2:")
    print(r2)
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(R.columns) + [u'类别数目']  # 重命名表头
    print("r:")
    print(r)

    # 详细输出原始数据及其类别
    r = pd.concat([R, pd.Series(model.labels_, index=R.index)], axis=1)  # 详细输出每个样本对应的类别
    r.columns = list(R.columns) + [u'聚类类别']  # 重命名表头
    r.to_csv("./output_data/RFAvg_day_labels.csv")  # 保存结果

    def density_plot(data):  # 自定义作图函数
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
        [p[i].set_ylabel(u'密度') for i in range(k)]
        plt.legend()
        return plt

    pic_output = ''  # 概率密度图文件名前缀
    for i in tqdm(range(k),desc='聚类图片保存中...'):
        density_plot(R[r[u'聚类类别'] == i]).savefig(u'./output_data/%s%s.png' % (pic_output, i))

    nums, SSE = test_Kmeans_nclusters(data_zs)

    # 画图，通过观察SSE与k的取值尝试找出合适的k值,SSE是簇内误方差方法，
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['font.size'] = 12.0
    plt.rcParams['axes.unicode_minus'] = False
    # 使用ggplot的绘图风格
    plt.style.use('ggplot')
    ## 绘图观测SSE与簇个数的关系
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, SSE, marker="+")
    ax.set_xlabel("n_clusters", fontsize=18)
    ax.set_ylabel("SSE", fontsize=18)
    fig.suptitle("KMeans", fontsize=20)
    plt.savefig('./output_data/分类效果图1.png')
    plt.show()


# 画出最好的分类
    kmodel = KMeans(n_clusters=4, n_jobs=4)
    kmodel.fit(data_zs)
    # 简单打印结果
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
    # 所有簇中心坐标值中最大值和最小值
    max = r2.values.max()
    min = r2.values.min()
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(data_zs.columns) + [u'类别数目']  # 重命名表头

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    center_num = r.values
    feature = ['F', 'R', 'avg_day']
    N = len(feature)
    for i, v in tqdm(enumerate(center_num),desc='画雷达图中...'):
        # 设置雷达图的角度，用于平分切开一个圆面
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # 为了使雷达图一圈封闭起来，需要下面的步骤
        center = np.concatenate((v[:-1], [v[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        # 绘制折线图
        ax.plot(angles, center, 'o-', linewidth=2, label="第%d簇人群,%d人" % (i + 1, v[-1]))
        # 填充颜色
        ax.fill(angles, center, alpha=0.25)
        # 添加每个特征的标签
        ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=15)
        # 设置雷达图的范围
        ax.set_ylim(min - 0.1, max + 0.1)
        # 添加标题
        plt.title('客户群特征分析图', fontsize=20)
        # 添加网格线
        ax.grid(True)
        # 设置图例
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1, fancybox=True, shadow=True)

    # 显示图形
    plt.savefig('./output_data/分类效果图2.png')
    plt.show()


    print('聚类分析完成！相关文件请到output_data中查找！')



if __name__ == '__main__':
    #记录时间


    start_time = time.time()

    # p = mp.Process(target=get_data1(), args=())
    # p.start()

    data1 = get_data1()
    data_avgday = process_data(data1)
    # print(data_avgday)

    data2 = get_data2()
    data_RF = process_data2(data2)
    # print(data_RF)

    data1_data2 = cat_data1_data2(data_RF,data_avgday)
    # print(data1_data2)

    main(data1_data2)

    time_dif = get_time_dif(start_time)
    print("程序整个运行时间：", time_dif)



