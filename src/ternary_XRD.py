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
from pyxrd.plotter import plot_ternary_diagram, rotate_phase_index, plot_xrd_on_ternary_line
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
top_directory = "data/GeSbSe_XRD/20240106_GeSbSn_100nm_300C_1h"
csv_file = 'data/GeSbSe_XRD/20240106_GeSbSn_100nm_300C_1h/GeSbSn_300C.xlsx'
title='GeSbSn_300C' #图片标题名和存储名

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
    ordered_custom_order[latter] = 'GeSn (Fd-3m)'
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
    ordered_custom_order.append('GeSn (Fd-3m) + Sn (I4_1/amd)')
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [148]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [161, 162, 146, 195, 146, 164]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [166]
    latter = 6
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    ordered_custom_order.append('Sb (R-3m) + Sb (P6_3/mmc)')
    point_Num = [0, 1, 2, 3, 4, 38, 37, 36, 35, 39,
                 40, 41, 73, 72, 71, 74, 75, 76, 103, 104,
                 105, 131]
    latter = 7
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    former = 2
    latter = 2
    ordered_custom_order[latter] = 'SnSb (R-3m) + Sb (P6_3/mmc)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    ordered_custom_order.append('SnSb (R-3m)')
    point_Num = [108, 128, 135, 151, 109, 127, 136, 150, 171, 99,
                 100, 110, 97, 111, 82, 125, 81, 66, 64, 67, 49,
                 137, 112, 80, 98, 65, 50, 62]
    latter = 8
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    ordered_custom_order.append('Sb (P6_3/mmc) + Sn (I4/mmm) + SnSb (R-3m)')
    point_Num = [12, 11, 10, 27, 26, 25]
    latter = 9
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    ordered_custom_order.append('SnSb (R-3m) + Sn (I4/mmm) + SnSb (I4_1/amd)')
    point_Num = [25, 14, 13, 52, 23]
    latter = 10
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    ordered_custom_order.append('Sn (I4/mmm) + SnSb (I4_1/amd)')
    point_Num = [24, 15, 16, 17, 22, 54, 53, 59, 60, 18, 21, 55]
    latter = 11
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [58, 88, 91, 117, 120]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    ordered_custom_order.append('SnSb (R-3m) + SnSb (I4_1/amd)')
    point_Num = [85]
    latter = 12
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [126, 51]
    latter = 0
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
    ordered_custom_order[latter] = 'Sb (R-3m) + Sb (P6_3/mmc)'
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

    point_Num = [209, 208, 204, 195, 194, 182, 203]
    latter = 3
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
    ordered_custom_order[latter] = 'Sb (R-3m) + SbSe (Pnma)'
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
    point_Num = [0, 1, 2, 3, 38, 37, 36, 35]
    latter = 2
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

    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

    point_Num = [172, 171, 170, 169, 168, 156, 157, 158, 98, 51]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [159, 161, 149, 152, 134, 128, 130, 100, 101, 102, 103, 74, 75, 76, 39, 40, 41, 42, 71, 70, 69, 68, 67,
                 65, 45, 46, 47, 48, 49, 62, 61, 86, 60, 52, 53, 54, 22, 44, 78, 79, 81, 80]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

    point_Num = [105]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'SbSeGe_150C':

    former = 3
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 2
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [190, 189, 188, 187, 186, 174, 175, 176, 177, 178, 179, 181, 173, 172]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    ordered_custom_order.append('Sb (R-3m) + SbSe (Pnma)')
    point_Num = [206, 200, 199, 189, 188, 174, 173, 205, 201, 198,
                 190, 175, 172, 187, 202, 197, 191, 186, 176, 196,
                 192, 185, 177, 193, 184, 178, 179]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == "SbSeGe_300C_new":

    former = 2
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 2
    latter = 0
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 1
    latter = 1
    ordered_custom_order[latter] = 'SbSe (Pnma)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    former = 2
    latter = 2
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')
    point_Num = [177]
    latter = 3
    ordered_custom_order[latter] = 'Sb (R-3m) + SbSe (Pnma)'
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [74, 11, 20, 107, 109]
    latter = 0
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [133, 134, 153, 156, 151, 152, 127, 129, 130, 131, 105, 172, 157, 150, 136, 126, 137,
                 138]
    latter = 1
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [155, 154, 158, 176, 209, 208, 207,
                 206, 205, 204, 200, 201, 202, 199,
                 198, 197, 196, 192, 191, 190, 189,
                 188, 187, 186, 185, 174, 175, 173,
                 171, 170, 159, 149]
    latter = 3
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')
    point_Num = [162, 163, 148, 147, 146, 145]
    latter = 2
    labels, ordered_custom_order = fb.exchange3(labels, point_Num, ordered_custom_order, latter, others='True')

if title == 'SbSeGe_as':
    former = 2
    latter = 1
    ordered_custom_order[latter] = 'Sb (R-3m)'
    labels, ordered_custom_order = fb.exchange2(labels, former, ordered_custom_order, latter=latter, others='True')

print('labels:', labels)
print('ordered_custom_order:', ordered_custom_order)

# 自定义峰
#printed_peak = 7 #索引必须在peaks_name范围内,最小值为1
#labels, ordered_custom_order = fb.every_peak(save_ori_labels, judge_array, peaks_name, printed_peak)
##############################peaks聚类的自定义区间##########################

##############################绘图及交互过程##########################
fig, ax = plt.subplots(figsize = (10, 6))
heatmap,tax = fb.plot_tri(ax,element3=element3,labels = labels,title= title,terlinenum = terlinenum, ordered_custom_order = ordered_custom_order)
#heatmap,tax = fb.plot_tri_con(ax,element3='Sn',labels = labels,title='GeSbSn_as',leveldown = 0,levelup = 3.5,terlinenum = terlinenum)
#tax.show()  # pycharm不支持show，插值连续图，但感觉不好看
#plt.savefig(save_path+title+'.png',dpi = 600) #保存图的命令
'''
plot_ternary_diagram(phase_type=ordered_custom_order,
                     phase_index=rotate_phase_index(labels),
                     labels=['Se', 'Ge', 'Sb'],
                     title=title,
                     color={
                         'amorphous': '#5EB89D',
                         'SnSb (R-3m)': '#D1363C',
                         'Sb (R-3m)': '#237AA6',
                         'SnSb (I4_1/amd)': '#E89C3D',
                         'Sn (I4_1/amd)': '#BD448E',
                         'GeSn (Fd-3m)': '#43624F',
                         'GeSe (Pnma)': '#40DB59',
                         'Se (P3121)': '#DB7335',
                         'SbSe (Pnma)': '#F56056'
                     },
                     if_show=False,
                     if_save=True,
                     if_legend=True,
                     rotation={
                         'GeSn (Fd-3m) + Sn (I4_1/amd)': 300,
                         'Sb (R-3m) + SnSb (I4_1/amd)': 0,
                         'Sb (R-3m) + SbSe (Pnma)': -90
                     }
                     )
'''
# 将点击事件绑定到散点图上,创建带有绑定参数的新函数，如果没有相应directory隐去下面部分
plt.ion()
on_pick_with_params = partial(fb.on_pick, directory=directory, angle_center=angle_center, r=r, sigma=sigma, window_size=window_size)
fig.canvas.mpl_connect('pick_event', on_pick_with_params)
plt.ioff()
plt.show()
#fig.savefig('SbSnGe-150C.png',dpi=600)
'''
plot_xrd_on_ternary_line(xrd_file_dir=directory,
                         start_point=(0, 100, 0),
                         end_point=(40, 0, 60),
                         detect_radius=1.2,
                         baseline_index=91,
                         v_margin=0.4, plot_peaks=False, factor=0.1, window=11, lam=100)
'''
##############################绘图及交互过程##########################
'''
import pickle
save_data = {'phase_index': labels, 'phase_type': ordered_custom_order}
with open(f'data/plot/{title}.pkl', 'wb') as f:
    pickle.dump(save_data, f)
'''
