# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:00:50 2024

@author: asus
本文件需要额外引用funda_XRD中的若干函数
仅用于三元成分坐标表示的情况，需要给定坐标和相应的csv(excel)峰值聚类情况
将输出高斯拟合模拟的峰值类别情况，以及五元分布，如果给定相应参数（原数据），可以完成图像交互
"""
import os
import sys
# 将当前路径的父文件夹添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Don't check this file with pylint and flake8
# pylint: disable-all
# flake8: noqa

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import src.funda_XRD as fb
from functools import partial
from pyxrd.plotter import plot_ternary_diagram, rotate_phase_index
plt.rc('font',family='Times New Roman')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

##############################基础变量及路径修改##########################
# 1. 导入CSV文件（非晶类型）
top_directory = "data/SbSeGe_XRD/20240418_SbSeGe_100nm_300C_1h"
csv_file = 'data/SbSeGe_XRD/20240418_SbSeGe_100nm_300C_1h/SbSeGe_300C.xlsx'
title='SbSeGe_300C' #图片标题名和存储名

df_peaks = pd.read_excel(csv_file)
#df_peaks = pd.read_csv(csv_file)
terlinenum = 20 #底面的点数，210个点对应20
#element3='Sn'
element3 = fb.findelement3(csv_file) #第三个元素的名称
save_path = 'figure/' #存储路径
# 指定包含.xy文件的目录,作对应单点图需要的(交互)
directory = [top_directory + "/22-analyze/",
             top_directory + "/46-analyze/"]
angle_center = [22, 46] #你有多少组的XRD_data

r = 5
sigma=20
window_size=50 #预处理的三个参数
#ordered_custom_order = []  #如果不给定分类信息则隐去该行
peaks_name = ['Sb','Sb','Sn','Sn$_4$Sb','Sn','Ge$_{11}$Sn$_{14}$','Sb','SbSn', 'Sb','Sb','Sn','Sn$_4$Sb','Sn','Ge$_{11}$Sn$_{14}$','Sb','SbSn'] #如果不给定峰的信息则隐去该行
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
#fb.plot_XRD_spectrum(xrd_spectra, peak_positions, peak_intensities, fig_labels,x_min=20, x_max=65)
##############################XRD模拟峰位图的绘制##########################

##############################peaks聚类的自定义区间##########################
'''
# GeSbSn-as
peaks1 = np.array([1])
peaks2 = np.array([1,2])

# GeSbSn-150C #11为amorphous
peaks1 = np.array([1])
peaks2 = np.array([1,2,3,4,5,6,12,13])
peaks3 = np.array([7,8,9,10,14,15])
peaks4 = np.array([6,8,9,15,16])
peaks5 = np.array([1,2,3,4,6,7,8,9,10,16])
peaks6 = np.array([1,2,13])
peaks7 = np.array([4,5])
peaks8 = np.array([8,10])
peaks9 = np.array([1])

labels = fb.exchange(labels, 11, 0)
labels = fb.exchange(labels, 2, 1)
labels = fb.exchange(labels, 3, 1)
labels = fb.exchange(labels, 4, 2)
labels = fb.exchange(labels, 5, 3)
labels = fb.exchange(labels, 6, 4)
labels = fb.exchange(labels, 7, 5)
labels = fb.exchange(labels, 8, 6)
labels = fb.exchange(labels, 9, 6)
labels = fb.exchange(labels, 10, 5)
labels = fb.exchange(labels, 12, 7)
labels = fb.exchange(labels, 13, 7)
labels = fb.exchange(labels, 14, 8)
labels = fb.exchange(labels, 15, 9)
labels = fb.exchange(labels, 16, 10)


# GeSbSn-300C #16为amorphous
peaks1 = np.array([1,2,3])
peaks2 = np.array([1,2,3,4,5,6,12,13])
peaks3 = np.array([7,8,9,10,14,15])
peaks4 = np.array([6,8,9,15,16])
peaks5 = np.array([9,10,12])#
peaks6 = np.array([1,2,13])
peaks7 = np.array([4,5])
peaks8 = np.array([8,10])
peaks9 = np.array([10,11,12])#
peaks10 = np.array([22,23,24,27,28,31])#
peaks11 = np.array([1])
peaks12 = np.array([25,30])#

labels = fb.exchange(labels, 2, 1)
labels = fb.exchange(labels, 19, 1)
labels = fb.exchange(labels, 21, 1)
labels = fb.exchange(labels, 3, 2)
labels = fb.exchange(labels, 4, 2)
labels = fb.exchange(labels, 5, 2)
labels = fb.exchange(labels, 6, 3)
labels = fb.exchange(labels, 7, 4)
labels = fb.exchange(labels, 8, 5)
labels = fb.exchange(labels, 9, 6)
labels = fb.exchange(labels, 14, 6)
labels = fb.exchange(labels, [10,20], 7)
labels = fb.exchange(labels, 11, 8)
labels = fb.exchange(labels, 12, 9)
labels = fb.exchange(labels, [13,27,22,24], 10)
labels = fb.exchange(labels, [15,30], 11)
labels = fb.exchange(labels, 16, 12)# 16-25-12-0
labels = fb.exchange(labels, 25, 12)# 16-25-12-0
labels = fb.exchange(labels, [17,18,29], 13)
labels = fb.exchange(labels, [23,31], 14)
labels = fb.exchange(labels, [26,28], 15)

# GeSbSe-150C 
peaks1 = np.array([1,7,8])
peaks2 = np.array([1,3,4,5,6])
peaks3 = np.array([1,2,4,5,7])
peaks4 = np.array([1,2,5,6,7])

labels = fb.exchange(labels, [7,8], 2)
labels = fb.exchange(labels, [4,5,6], 1) #1必须在1
labels = fb.exchange(labels, 3, 3)

# GeSbSe-300C
peaks1 = np.array([1,2,3,6,9])
peaks2 = np.array([6,7,8])
peaks3 = np.array([1,2,5,6])
peaks4 = np.array([1,2,3,4,5,6,8])

labels = fb.exchange(labels, [2,3,4,5,9], 1)
labels = fb.exchange(labels, [6,8], 2)
labels = fb.exchange(labels, 7, 3)

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
#labels = fb.exchange(labels, peaks4, 1, others = 'False')
'''
# 自定义类型
if title == 'GeSbSn_150C':
    
    former = 5
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 2
    latter = 2
    ordered_custom_order[latter] = 'SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 1
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [38]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 5
    latter = 5
    ordered_custom_order[latter] = 'Sn (I4_1/amd) + SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 6
    latter = 6
    ordered_custom_order[latter] = 'SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 7
    latter = 7
    ordered_custom_order[latter] = 'Sn (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [19, 20]
    latter = 7
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 4
    latter = 5
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [16]
    latter = 4
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [23, 24, 52, 54, 55, 56, 57, 58, 89, 142,141,123,121,120,114,115,116,117,118,91,92,93,94,95,90,84,85,63,62,51,52, 122]
    latter = 5
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'Sb (R-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [36, 37, 73, 72, 41, 42, 71]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [107, 108, 128,99,111,96,82,46,28,7]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [37, 36, 40, 39, 73, 71, 72, 75, 74, 104, 106, 105, 103, 41]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 2
    latter = 2
    ordered_custom_order[latter] = 'SnSb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 3
    latter = 2
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [17, 18, 56, 55, 57, 89, 90, 118, 91, 88, 58]
    latter = 5
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'SnSb (R-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [51, 63, 24, 52, 62, 84, 83, 96]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [113]
    latter = 4
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'GeSbSn_300C':

    former = 10
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 9
    latter = 9
    ordered_custom_order[latter] = 'SnSb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 12
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 2
    latter = 2
    ordered_custom_order[latter] = 'Sb (R-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 12
    latter = 2
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = [3,11]
    latter = 10
    ordered_custom_order[latter] = 'SnSb (Pm-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 3
    latter = 9
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 9
    latter = 8
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 3
    latter = 3
    ordered_custom_order[latter] = 'SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 6
    latter = 6
    ordered_custom_order[latter] = 'SnSb (I4_1/amd) + Ge (Fd-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 9
    latter = 6
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 5
    latter = 5
    ordered_custom_order[latter] = 'SnSb (I4_1/amd) + Sn (Fd-3m) + SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 4
    latter = 5
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [207,208,205,200,201,202,203,199,198,197,196,195,189,190,192,193,184,185,186,187,188,175,176,177,178,180,167,168,169,170,172,173,155,156,157,158,160,162]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    former = 8
    latter = 8
    ordered_custom_order[latter] = 'Ge (Fd-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [194,182]
    latter = 8
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [171]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [163]
    latter = 5
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'SnSb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 5
    latter = 5
    ordered_custom_order[latter] = 'SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [73, 72, 41, 71, 74, 104, 131, 103, 106, 132, 130, 75, 105, 76, 102, 107]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [34, 43, 6, 70, 77, 44, 45, 32, 7, 31, 30, 101, 78, 100, 79, 68, 8, 9, 10, 29, 11, 28, 48]
    latter = 2
    ordered_custom_order[latter] = 'SnSb (R-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [129, 154, 153, 152, 133, 134, 84, 83]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 7
    latter = 3
    ordered_custom_order[latter] = 'SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [127, 150, 136, 137, 125, 109, 110, 98, 80, 99, 49, 82, 112, 111, 26]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [25, 85, 14]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 4
    latter = 3
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 4
    latter = 3
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [92, 87, 59, 60, 164, 121, 93, 115, 116, 118]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [181, 195, 166, 165, 142, 141, 162, 146, 148, 161]
    latter = 5
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [164]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 4
    latter = 4
    ordered_custom_order[latter] = 'Sn (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [144, 119, 118, 91, 88, 58, 55, 21, 18, 120, 117]
    latter = 4
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [164, 163, 143, 144, 142, 141]
    latter = 6
    ordered_custom_order.append('Ge (Fd-3m) + Sn (I4_1/amd)')
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'SbSnGe_150C':

    former = 5
    latter = 8
    ordered_custom_order[latter] = 'SnSb (Pm-3m) + Sn (Fd-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 4
    latter = 5
    ordered_custom_order[latter] = 'SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'Sn (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [141, 164]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 1
    latter = 3
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [18, 19, 20, 22, 11, 12, 14, 24, 5, 6, 30, 44, 32, 37]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [172, 130, 131, 153, 132, 152, 151, 150, 137, 147, 158, 170, 169, 160, 148, 133, 168, 177, 188, 187, 186]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [0,1,2,3]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [35, 36, 39, 40]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 2
    latter = 2
    ordered_custom_order[latter] = 'SnSb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [174, 172, 171, 170, 175, 187, 177, 160]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    #point_Num = [188, 187, 186, 176, 174, 177, 175, 168, 169, 161, 159, 170, 171, 160, 147, 148, 149, 172]
    #latter = 3
    #labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'SbSnGe_300C':

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'SnSb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 6
    latter = 6
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 1
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m) + SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 5
    latter = 5
    ordered_custom_order[latter] = 'Sb (R-3m) + SnSb (I4_1/amd)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 8
    latter = 8
    ordered_custom_order[latter] = 'SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 7
    latter = 5
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 7
    latter = 5
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 4
    latter = 2
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 3
    latter = 3
    ordered_custom_order[latter] = 'SnSb (Pm-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [127]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [0,1,2]
    latter = 6
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 3
    latter = 2
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [191, 192, 197, 201, 184, 185, 186, 187, 175, 176, 177, 178, 179, 180, 166, 167, 168, 169, 179, 170, 171, 172, 173, 163, 162]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 1
    latter = 2
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [0, 1, 2, 37, 36, 35]
    latter = 4
    ordered_custom_order.append('Sn (I4_1/amd)')
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [126, 137]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == "GeSbSe_150C":

    former = 1
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 3
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 4
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 2
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 2
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

if title == 'GeSbSe_300C':
    
    former = 1
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 4
    latter = 4
    ordered_custom_order[latter] = 'Sb (R-3m) + Se (R-3)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 3
    latter = 4
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [8, 9, 7, 10, 11, 12, 6, 5, 4, 30, 29, 28]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    former = 2
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 3
    latter = 1
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [49]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'SbSeGe_300C':

    former = 2
    latter = 1
    ordered_custom_order[latter] = 'GeSe (Pnma) + Se (P3121)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    former = 4
    latter = 4
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [177, 176, 174, 173]
    latter = 4
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    former = 3
    latter = 2
    ordered_custom_order[latter] = 'SbSe (Pnma)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

print('labels:', labels)
print('ordered_custom_order:', ordered_custom_order)

# 自定义峰
#printed_peak = 7 #索引必须在peaks_name范围内,最小值为1
#labels, ordered_custom_order = fb.every_peak(save_ori_labels, judge_array, peaks_name, printed_peak)
##############################peaks聚类的自定义区间##########################

##############################绘图及交互过程##########################
#fig, ax = plt.subplots(figsize = (10, 6))
#heatmap,tax = fb.plot_tri(ax,element3=element3,labels = labels,title= title,terlinenum = terlinenum, ordered_custom_order = ordered_custom_order)
#heatmap,tax = fb.plot_tri_con(ax,element3='Sn',labels = labels,title='GeSbSn_as',leveldown = 0,levelup = 3.5,terlinenum = terlinenum)
#tax.show()  # pycharm不支持show，插值连续图，但感觉不好看
#plt.savefig(save_path+title+'.png',dpi = 600) #保存图的命令

plot_ternary_diagram(phase_type=ordered_custom_order,
                     phase_index=labels,
                     labels=['Se', 'Ge', 'Sb'],
                     title=title,
                     color={
                         'amorphous': '#5EB89D',
                         'SnSb (R-3m)': '#D1363C',
                         'Sb (R-3m)': '#237AA6',
                         'SnSb (I4_1/amd)': '#E89C3D',
                         'Sn (I4_1/amd)': '#BD448E',
                         'Ge (Fd-3m)': '#43624F',
                         'Se (R-3)': '#F56056',
                         'GeSe (Pnma)': '#40DB59',
                         'Se (P3121)': '#DB7335',
                         'SbSe (Pnma)': '#9CDC3A'
                     },
                     if_show=True,
                     if_save=True,
                     if_legend=True,
                     rotation={
                         'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                         'Sb (R-3m) + SnSb (I4_1/amd)': 0
                     }
                     )

# 将点击事件绑定到散点图上,创建带有绑定参数的新函数，如果没有相应directory隐去下面部分
plt.ion()
on_pick_with_params = partial(fb.on_pick, directory=directory, angle_center=angle_center, r=r, sigma=sigma, window_size=window_size)
fig.canvas.mpl_connect('pick_event', on_pick_with_params)
plt.ioff()
#fig.savefig('SbSnGe-150C.png',dpi=600)
plt.show()
##############################绘图及交互过程##########################