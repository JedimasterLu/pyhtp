# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:59:49 2024
基础函数在funda_XRD.py文件里
本文件给定示例角度类，找到示例角度附近正负0.5°的峰作为1类
示例角度类就是given_angles，从peak_clustering的全局观察结果+层次聚类结果可以进行定义
附近正负0.5°这个值可以修改，在funda_XRD.py文件里的process_files()的 angle_tolerance=0.5这里修改
@author: asus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
#from sklearn.preprocessing import StandardScaler
import funda_XRD as fb

# 指定给定角度列表和目录
#given_angles = np.array([32.06, 44.95])
#given_angles = np.array([34.93, 40.12, 41.70, 44.10, 45.05, 48.30])  # 示例给定角度
#given_angles = np.array([35.07, 40.06, 41.65, 44.15, 44.95, 48.08, 51.65])  # 示例给定角度
#given_angles = np.array([23.64, 29.13, 30.78, 32.08])  # 示例给定角度
#given_angles = np.array([23.61, 29.06, 31.08, 32.04, 33.48])  # 示例给定角度
given_angles = np.array([23.79, 25.37, 31.87, 34.70, 48.45, 51.95])

# 指定包含.xy文件的目录和文件夹名
"""
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20240106_GeSbSn_100nm_300C_1h\\46-analyze'
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20240108_GeSbSn_100nm_150C_1h\\22-analyze'
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSn_100nm\\46-analyze'
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSe_100nm_150C_1h\\22-analyze'
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSe_100nm_300C_1h\\46-analyze'
'F:\\学术研究\\数据存储\\Ge-Sb-X系相变材料\\20240114_XRD\\20231114_GeSbSe_100nm\\46-analyze
"""
file_directory = 'data/SbSeGe_XRD/20240719_SbSeGe_100nm_300C_1h_new/'
file = '46-analyze'
directory = file_directory + file

# 指定保存路径及名称
save_path = file_directory + '%s.xlsx'%file

# 指定预处理的三个参数
r=5
sigma=20
window_size=50

##############################以下是计算过程，无需修改##########################
#输出模板
oriexcel, real_columns = fb.excel_pattern(given_angles=given_angles, directory = directory)
# 处理文件并收集数据,保存为Excel文件
fb.process_files(directory, given_angles, oriexcel, real_columns, save_path = save_path, r=r, sigma=sigma, window_size=window_size)

