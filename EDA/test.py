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

	#获取数据
	def GetData():
		updata = st.file_uploader('上传数据地址',type=['csv','xls','xlsx'])
		if updata is not None:
			try:
				excel_reader = pd.ExcelFile(updata)
				sheet_names = excel_reader.sheet_names

				if len(sheet_names) > 1:
					title = st.selectbox('所上传的数据有多个表格，选择其中要读取的数据表格为', sheet_names)
					st.write('选择打开的表格为：'+title)

				return(pd.read_excel(updata,sheet_name=title,header=1))
			except:
				st.warning('csv类型文件还有问题，解决中...')
				return(pd.read_csv(updata,engine='python',header=0,skiprows=1))

	#为基本数据分析模块
	def Model1():
		# 第一个模块
		if st.sidebar.checkbox('1.基本分析数据'):
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
				col = st.sidebar.selectbox('选择列看其中唯一属性', EA.columns)
				st.subheader('唯一值和频数')
				st.write(EA.info2(col))

	#数据可视化模块
	def Model2():
		# 第二个模块
		if st.sidebar.checkbox('2.数据可视化'):

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

			if st.sidebar.checkbox('其他类型画图'):
				st.subheader('其他类型画图')
				st.write('完善中...')

	try:
		df = GetData()
		EA = ExploratoryAnalysis(df)
		st.success('文件上传成功')
		st.sidebar.title('数据分析菜单')
		Model1()
		Model2()
		st.sidebar.title('作者：cloneman')
	except:
		st.error('请上传csv或者xls格式的文件！')

if __name__ == "__main__":
	st.title("数据探索分析软件")

	selected = st.selectbox('选择进行数据分析类型',(None,'传统数据分析','深度学习分析'))

	if selected == '传统数据分析':
		st.subheader('拖动或者上传文件到指定的框内就可以进行数据分析')
		main()

	if selected =='深度学习分析':
		st.info('开发中......')

	if selected ==None:
		st.warning('请选择想使用的方法')

