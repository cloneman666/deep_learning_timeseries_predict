#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020-10-27 08:33
# @Author  : Cloneman
# @Email   : 1171836398@qq.com
# @File    : test.py
# @Software: PyCharm

from ExploratoryAnalysis import ExploratoryAnalysis
import streamlit as st
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False) #先不显示画图警告

def main():


	def GetData():
		updata = st.file_uploader('上传数据地址',type=['csv','xls'])
		if updata is not None:
			try:
				return(pd.read_excel(updata,header=1,sheet_name=2))
			except:
				return(pd.read_csv(updata))


	df = GetData()
	try:
		EA = ExploratoryAnalysis(df)
		st.success('文件上传成功')
		st.sidebar.title('数据分析菜单')

		st.sidebar.subheader('分析数据')
		if st.sidebar.checkbox('使用基本探索分析选项'):
			if st.sidebar.checkbox('显示前5行'):
				st.subheader('展示数据前5行：')
				st.write(df.head())
			if st.sidebar.checkbox('数据信息（information）'):
				st.subheader('数据的基本信息如下：')
				st.text(df.info())
			if st.sidebar.checkbox('数据描述（describe）'):
				st.subheader('数据的描述信息如下：')
				st.write(df.describe())
			if st.sidebar.checkbox('是否有空值'):
				st.subheader('空值信息如下：')
				st.write(df.isnull().sum())
			if st.sidebar.checkbox('唯一值和频数'):
				col = st.sidebar.selectbox('选择列看其中唯一属性',EA.columns)
				st.subheader('唯一值和频数')
				st.write(EA.info2(col))

		st.sidebar.subheader('数据可视化')
		if st.sidebar.checkbox('使用数据画图方法'):
			if st.sidebar.checkbox('平面图'):

				if st.sidebar.checkbox('计数画图'):
					st.subheader('计数画图为：')
					column_count_plot = st.sidebar.selectbox("选择列", EA.columns)
					hue_opt = st.sidebar.selectbox("选择分类的变量",
												   EA.columns.insert(0, None))
					if st.checkbox('展示计数画图'):
						fig = EA.CountPlot(column_count_plot, hue_opt)
						st.pyplot()

				if st.sidebar.checkbox('分布式画图'):
					st.subheader('分布式画图为：')
					column_dist_plot = st.sidebar.selectbox("选择列",
															EA.numerical_columns)
					if st.checkbox('展示分布式画图'):
						fig = EA.DistPlot(column_dist_plot)
						st.pyplot()







	except:
		st.error('请上传csv或者xls格式的文件！')

if __name__ == "__main__":
	st.title("数据探索分析软件")
	main()