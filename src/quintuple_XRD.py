# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:00:50 2024

@author: asus
本文件需要额外引用funda_XRD中的若干函数
仅用于五元或者直接用坐标而不是成分坐标表示的情况，需要给定坐标和相应的csv(excel)峰值聚类情况
将输出高斯拟合模拟的峰值类别情况，以及五元分布，如果给定相应参数（原数据），可以完成图像交互
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import src.funda_XRD as fb
from functools import partial
plt.rc('font',family='Times New Roman')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

##############################基础变量及路径修改##########################
# 导入CSV文件（聚类结果）
#csv_file = 'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSn_100nm\\GeSbSn_as.xlsx'
csv_file = 'F:\\学术研究\\数据存储\\20230901-NiCrAl磁控溅射\\XRD图谱\\MI3_popt_all.csv'
#df_peaks = pd.read_excel(csv_file)
df_peaks = pd.read_csv(csv_file)
# 导入五元坐标及绘制散点图
arrayPath = "F:\\学术研究\\数据存储\\20230901-NiCrAl磁控溅射\\"
arraydata = pd.read_excel(arrayPath + 'Magnetron_sputtering.xlsx')
# 保存路径及文件名
title='NiCrAlNbMo_1_700C_2h.png' #图片标题名和存储名
save_path = 'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSe_100nm_300C_1h\\' #存储路径
# 指定包含.xy文件的目录,作对应单点图需要的
directory = ["F:\\学术研究\\数据存储\\20230901-NiCrAl磁控溅射\\XRD图谱\\NiCrAlNbMo1-40-FZY\\"]
angle_center = [40] #你有多少组的XRD_data

r = 5
sigma=20
window_size=50 #预处理的三个参数
#ordered_custom_order = []  #如果不给定分类信息则隐去该行
peaks_name = ['1','2','3','4','5','6'] #如果不给定峰的信息则隐去该行
##############################基础变量及路径修改##########################

##############################以下是定义全局变量过程，无需修改##########################
# 由给出参数计算全局变量
column_num = df_peaks.shape[1]
array_num = df_peaks.shape[0]
highs1_num = int((column_num-2)/2 + 2) #找到highs1的位置
angles_num = int((column_num-2)/2) #共有多少个角度
judge = np.array(df_peaks.iloc[:,highs1_num:column_num])*10  #因为小于0.1不算峰，0.1*10=1
judge = judge.astype(int)
judge[np.where(judge != 0)] = 1 #后续用以判断数组，判断组合是否重复，便于比较切为0 1
##############################以上是定义全局变量过程，无需修改##########################

#########自动程序，配合峰信息##########
xrd_spectra, peak_positions, peak_intensities, fig_labels, labels, label_name, judge_array = fb.xrd_spectra(df_peaks, angles_num, highs1_num, column_num, judge, peaks_name)
save_ori_labels = copy.deepcopy(labels)
labels, ordered_custom_order = fb.auto_exchange(labels, label_name)
#########自动程序，配合峰信息##########

#########常规程序，无需配合峰信息##########
#xrd_spectra, peak_positions, peak_intensities, fig_labels, labels = fb.xrd_spectra(df_peaks, angles_num, highs1_num, column_num, judge)
#########常规程序，无需配合峰信息##########

##############################XRD模拟峰位图的绘制##########################
# XRD模拟峰位图的绘制
fb.plot_XRD_spectrum(xrd_spectra, peak_positions, peak_intensities, fig_labels,x_min=25, x_max=70)
##############################XRD模拟峰位图的绘制##########################

##############################peaks聚类的自定义区间##########################
'''
# NiCrAlNbMo-1-700C
peaks1 = np.array([2,3,4,6,8,10,11,12,14,15])
peaks2 = np.array([5,9,10,12])
peaks3 = np.array([5,9,10,12])
peaks4 = np.array([4,8,11,12,14,16,17,18])
peaks5 = np.array([1,3,5,6,7,8,9,10,11,12,18])
peaks6 = np.array([6,7,9,10,11,12,13,14,15,16,18])


# NiCrAlNbMo-3-700C
peaks1 = np.array([15,17])
peaks2 = np.array([1,2,4,5,6,7,8,10,12,16])
peaks3 = np.array([4,6,7,9,12,13,16,19,20])
peaks4 = np.array([7,9])
peaks5 = np.array([3,5,8,10,12,14,16,19,21,22])
peaks6 = np.array([6,14,16,18,19,20])
peaks7 = np.array([10,11,12,17,18,19,22])

#fb.exchange(labels, peaks5, 9, others = 'True') #others=True 表明其他峰位保留，为默认，否则其他峰位自动合成一个
labels = fb.exchange(labels, peaks7, 1, others = 'False')



fb.exchange(labels, 7, 6)
fb.exchange(labels, 9, 6)
fb.exchange(labels, 10, 6)
fb.exchange(labels, 11, 6)
fb.exchange(labels, 12, 6)
fb.exchange(labels, 13, 6)
fb.exchange(labels, 14, 6)
fb.exchange(labels, 15, 6)
fb.exchange(labels, 16, 6)
fb.exchange(labels, 18, 6)
'''

# 自定义类型
#former = 11
#labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=0, others='True')

# 自定义峰
#printed_peak = 1 #索引必须在peaks_name范围内,最小值为1
#labels, ordered_custom_order = fb.every_peak(save_ori_labels, judge_array, peaks_name, printed_peak)
##############################peaks聚类的自定义区间##########################

##############################绘图及交互过程##########################
# 绘制散点图
fig = fb.plot_quin(arraydata, labels, title, ordered_custom_order = ordered_custom_order)
plt.show()
#plt.savefig(save_path+title+'.png',dpi = 600) #保存图的命令

# 将点击事件绑定到散点图上,创建带有绑定参数的新函数，如果没有相应directory隐去下面部分
on_pick_with_params = partial(fb.on_pick, directory=directory, angle_center=angle_center, r=r, sigma=sigma, window_size=window_size)
fig.canvas.mpl_connect('pick_event', on_pick_with_params)
plt.show()
##############################绘图及交互过程##########################