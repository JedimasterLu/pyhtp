# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:35:14 2024
基础函数在funda_XRD.py文件里
本文件需要给定所有原数据.xy的文件夹作为全局分析
或者某一些原数据.xy的文件作为局部分析
总体上为作图（包含预处理前和后）
全局分析还会自动配合层次聚类告知你峰的具体数值（均值）
@author: asus
"""
#%%  全局分析(run current cell)

import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os
import matplotlib.pyplot as plt
import funda_XRD as fb

# 指定包含.xy文件的目录
directory = r"data\SbSeGe_XRD\20240418_SbSeGe_100nm_300C_1h\22-analyze/"
cluster_number = 10 #层次聚类数量
r=5
sigma=20
window_size=50 #预处理的三个参数

fb.plot_xy_files(directory)
all_peaks, labels = fb.observe_pre(directory, r=r, sigma=sigma, window_size=window_size)
fb.clustersort(all_peaks, number = cluster_number)

#%%  单点分析(run current cell) 
import numpy as np
import matplotlib.pyplot as plt
import src.funda_XRD as fb
# 指定包含.xy文件的目录
directory = r"zgm\20240106_GeSbSn_100nm_300C_1h\22-analyze/"
name = '22-172-0000_exported'
r = 5
sigma=20
window_size=50 #预处理的三个参数

#计算过程
data = np.loadtxt(directory + name + '.xy', encoding='utf-8', skiprows=1)
x = data[:, 0]  # 2θ角度
y = data[:, 1]  # 强度值
y_ori = data[:, 1]  # 初始强度值
y = fb.preprocess_data(x, y, r=r, sigma=sigma, window_size=window_size)
plt.plot(x, y, label='1')#去背底
plt.plot(x, y_ori, label='2')#原始数据
plt.show()

#%%  局部分析  #注意，暂时没有拟合合并突变点附近段
import numpy as np
import matplotlib.pyplot as plt
import src.funda_XRD as fb
# 指定包含.xy文件的目录
directory = ["zgm/20240106_GeSbSn_100nm_300C_1h/22-analyze/",
             "zgm/20240106_GeSbSn_100nm_300C_1h/46-analyze/"]
name = [['22-172-0000_exported','22-171-0000_exported','22-000-0000_exported'],
        ['46-172-0000_exported','46-171-0000_exported','46-000-0000_exported']]   #逻辑必须是相同角度的在同一个文件夹中，或者说name[i]中的文件要在同一个文件夹中,双中括号索引
angles = [22,46] #必须与name对应

r = 5
sigma=20
window_size=50 #预处理的三个参数
updown_gap= 2 #两个XRD谱之间的距离设置

fb.plot_merge_XRD(directory,name,angles,r,sigma,window_size,updown_gap)


# %%
