# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:37:26 2024
XRD自动分析流程的基础函数
@author: asus
"""
# Don't check this file with pylint and flake8
# pylint: disable-all
# flake8: noqa
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
#from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import curve_fit
import copy
import ternary
from scipy.interpolate import griddata
from functools import partial
plt.rc('font',family='Times New Roman')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels


#加载文件名下所有.xy文件的数据
def read_xy_file(file_path):
    """ 读取.xy文件并返回x和y的列表 """
    x, y = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
    return x, y

#由加载文件名下所有.xy文件的数据作图(初始图)
def plot_xy_files(directory,lim_high = 2):
    """ 读取指定目录下的所有.xy文件，并将它们绘制在同一张图上 """
    for filename in os.listdir(directory):
        if filename.endswith('.xy'):
            file_path = os.path.join(directory, filename)
            x, y = read_xy_file(file_path)
            plt.plot(x, y, label=filename)

    plt.xlabel('2theta')
    plt.ylabel('Idensity')
    plt.title('XRD_patterns')
    plt.ylim(0,lim_high)
    plt.show()

# 数据预处理
def preprocess_data(x, y, r=5, sigma=20, window_size=50):
    """ 应用高斯滤波和滚动球算法 """
    y_gaussian = gaussian_filter1d(y, sigma=sigma)
    y = uniform_filter1d(y_gaussian, size=window_size)

    # 滚球法处理数据
    y_bg = np.zeros_like(y)
    for i in range(len(x)):
        # 找出小球内的数据点
        mask = (x > x[i]-r) & (x < x[i]+r)
        y_ball = y[mask]
        # 计算背景信号
        y_bg[i] = np.mean(y_ball)
        # 减去背景信号
        y[i] = y[i] - y_bg[i]
        y[i] = (y[i] + abs(y[i]))/2 

    return y

# 函数来计算半高宽
def compute_fwhm(x, y, peak):
    half_max = y[peak] / 2
    # 确保使用整数索引
    peak = int(peak)
    # 使用插值找到半高宽的左右点
    try:
        left_idx = np.where(y[:peak] <= half_max)[0][-1]
        right_idx = np.where(y[peak:] <= half_max)[0][0] + peak
        fwhm = x[right_idx] - x[left_idx]
        #print(fwhm)
        return fwhm
    except IndexError:
        # 如果在数据边界内找不到半高宽，返回无穷大或其他标记值
        return np.inf

# 峰位检测，输出满足条件（半高宽）的峰位，threshold是相对峰高，fwhm_threshold对应了半峰宽，但转换为了theta的step数量（大概480对应2~3度），这里注意研究非晶时要仔细调整
def detect_peaks(x, y, threshold, fwhm_threshold=480):
    """ 峰值检测，考虑半高宽限制 """
    peaks, properties = find_peaks(y, prominence=threshold)
    selected_peaks = []
    count = 0
    for peak in peaks:
        if compute_fwhm(x, y, peak) < fwhm_threshold:
            # 计算峰前的背底强度
            peak_intensity = properties["prominences"][count]
            selected_peaks.append((x[peak], peak_intensity))
        count = count + 1
    return selected_peaks

def observe_pre(directory, r=5, sigma=20, window_size=50): #观测全部处理后的数据
    all_peaks = []
    labels = []
    plt.figure()
    for filename in os.listdir(directory):
        if filename.endswith('.xy'):
            file_path = os.path.join(directory, filename)
            x, y = read_xy_file(file_path)
            x = np.array(x)
            y = np.array(y)
            # 数据预处理
            y_processed = preprocess_data(x, y, r=r, sigma=sigma, window_size=window_size)
    
            # 峰值检测
            peaks = detect_peaks(x, y_processed, threshold=0.1)
            first_elements = np.array([x[0] for x in peaks])
            peaks = first_elements
            all_peaks.extend(peaks)
            labels.extend([filename] * len(peaks))
    
            # 绘制预处理后的数据
            plt.plot(x, y_processed, label=filename)
            plt.xlabel('2theta')
            plt.ylabel('Idensity')
            plt.title('XRD_patterns')
            #plt.ylim(0,2)
            #plt.legend()
            plt.show()
    return all_peaks, labels

def clustersort(all_peaks, number = 10): #层次聚类
    # 聚类分析
    # 1. 对所有峰值进行排序
    all_peaks.sort()
    
    # 2. 聚类
    clusters = []
    current_cluster = [all_peaks[0]]
    
    for peak in all_peaks[1:]:
        if peak - current_cluster[-1] <= 0.5:
            current_cluster.append(peak)
        else:
            clusters.append(current_cluster)
            current_cluster = [peak]
    
    # 添加最后一个聚类
    clusters.append(current_cluster)
    
    # 3. 计数并选择出现次数最多的10组
    cluster_counts = [(len(cluster), cluster) for cluster in clusters]
    cluster_counts.sort(reverse=True)
    top_clusters = cluster_counts[:10]
    clusters = np.array(clusters)
    # 打印结果
    for count, cluster in top_clusters:
        print(f"峰值：{np.average(cluster)}, 出现次数：{count}")

def excel_pattern(given_angles, directory): #输出csv的模板
    #定义初始矩阵
    len1 = len(os.listdir(directory))
    oriexcel = np.zeros((len1,2+2*np.shape(given_angles)[0]))
    real_columns=['No.', 'scanNum']
    
    for i in range(len1):
        oriexcel[i,0] = i
        oriexcel[i,1] = i
    
    for i in range(np.shape(given_angles)[0]):
        oriexcel[:,i+2] = given_angles[i]
        peak_label = 'peaks' + str(i)
        real_columns.append(peak_label)
        
    for i in range(np.shape(given_angles)[0]):
        peak_label = 'highs' + str(i)
        real_columns.append(peak_label)
    return oriexcel, real_columns

# 检查峰值角度是否与给定角度相差在0.5°以内，且满足（半高宽）条件，输出csv
def process_files(directory, given_angles, oriexcel, real_columns, angle_tolerance=0.5, fwhm_threshold=480, save_path = 'peak_data.xlsx', r=5, sigma=20, window_size=50):
    count1 = 0 #列
    for filename in os.listdir(directory):
        if filename.endswith('.xy'):
            file_path = os.path.join(directory, filename)
            x, y = read_xy_file(file_path)
            x = np.array(x)
            y = np.array(y)
            if len(x) != len(y):
                continue  # 如果 x 和 y 长度不同，跳过此文件
            y_processed = preprocess_data(x, y, r=r, sigma=sigma, window_size=window_size)        
            peaks = detect_peaks(x, y_processed, threshold=0.1, fwhm_threshold=fwhm_threshold)
            for peak_angle, peak_intensity in peaks:
                # 检查峰值角度是否与给定角度相差在0.5°以内
                count2 = 0 #行
                for angle in given_angles:
                    if abs(peak_angle - angle) <= angle_tolerance:
                        oriexcel[count1,count2+2] = peak_angle
                        oriexcel[count1,count2+2+np.shape(given_angles)[0]] = peak_intensity
                    count2 = count2+1
        count1 = count1+1
        #print(count1)
    df = pd.DataFrame(oriexcel, columns=real_columns)
    df.to_excel(save_path, index=False)

# 定义高斯函数（用于拟合）
def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

def mkdir(path):
    # 此函数自动创建文件夹路径
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)  
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False

# 选取需要的xrd_spectra
def xrd_spectra(df_peaks, angles_num, highs1_num, column_num, judge):
    xrd_spectra = []
    labels = np.zeros(len(df_peaks))
    labels = labels.astype(int) #用于分类的标记，0表示非晶，1到n表示不同的结晶类型
    labelstart = 1
    judge_array = np.zeros(angles_num)
    judge_array = judge_array.astype(int)
    boolflag = -1
    fig_labels = [] #作图显示的label
    
    # 找到不同的分类（不包含自动命名），建议不要轻易修改逻辑
    for i in range(0, len(df_peaks)):
        if (int(np.sum(df_peaks.iloc[i, highs1_num:column_num])*10) == 0):
            continue
            
        if boolflag == -1:
            for j in range(0,1):
                temp = judge[i] - judge_array
                if temp.any(): #第一次必定是新组合
                    judge_array = judge[i]
                    labels[i] = labelstart
                    fig_labels.append(labelstart)

        else:
            for j in range(0,np.shape(judge_array)[0]):
                temp = judge[i] - judge_array[j]
                #print(np.shape(judge_array)[0])
                if temp.any(): 
                    boolflag = boolflag#print('1')
                else: #所有都为0，说明是已经留存过的组合
                    boolflag = 1
                    labels[i] = j+1 #由于顺序一致的兼容性，可以用j直接来代替
                    break
            if boolflag != 1: #这个表示一类
                labelstart = labelstart+1
                judge_array = np.vstack([judge_array,judge[i]])
                labels[i] = labelstart
                fig_labels.append(labelstart)
                
        if boolflag == 1:
            boolflag = 0
            continue
        
        # 峰位（2Theta）和峰强度数据
        peak_positions = np.array(df_peaks.iloc[i, 2:highs1_num])  
        peak_intensities = np.array(df_peaks.iloc[i, highs1_num:column_num])  
        #peak_widths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 用你的峰宽度数据替换
        peak_widths = np.zeros(angles_num)+0.1 # 用你的峰宽度数据替换
    
        # 绘制XRD谱
        x = np.linspace(min(peak_positions) - 5, max(peak_positions) + 5, 1000)  # 生成X轴坐标

        # 初始化XRD图谱
        xrd_spectrum = np.zeros_like(x)
    
        # 遍历每个峰并添加到XRD图谱
        for position, intensity, width in zip(peak_positions, peak_intensities, peak_widths):
            # 初始化参数值，可以根据实际数据进行调整
            initial = [intensity, position, width]
    
            # 将高斯峰添加到XRD图谱
            xrd_spectrum += gaussian(x, *initial)
        if boolflag == -1:
            xrd_spectra = xrd_spectrum
            boolflag = 0
        else:
            xrd_spectra = np.vstack([xrd_spectra, xrd_spectrum])
    #print(np.shape(judge_array))
    return xrd_spectra, peak_positions, peak_intensities, fig_labels, labels

# 选取需要的xrd_spectra,加入peaks峰数据，多出传参peaks_name
def xrd_spectra(df_peaks, angles_num, highs1_num, column_num, judge, peaks_name):
    xrd_spectra = []
    labels = np.zeros(len(df_peaks))
    labels = labels.astype(int) #用于分类的标记，0表示非晶，1到n表示不同的结晶类型
    labelstart = 1
    judge_array = np.zeros(angles_num)
    judge_array = judge_array.astype(int)
    boolflag = -1
    fig_labels = [] #作图显示的label
    label_name = [] #和peak_name对应
    
    # 找到不同的分类（包含自动命名），建议不要轻易修改逻辑
    for i in range(0, len(df_peaks)):
        if (int(np.sum(df_peaks.iloc[i, highs1_num:column_num])*10) == 0):
            continue
            
        if boolflag == -1:
            for j in range(0,1):
                temp = judge[i] - judge_array
                if temp.any(): #第一次必定是新组合
                    judge_array = judge[i]
                    labels[i] = labelstart
                    fig_labels.append(labelstart)
            
                    name = [] #标签名自动计算
                    for k in range(0,np.shape(np.argwhere(judge[i] == 1))[0]):
                        a = peaks_name[np.argwhere(judge[i] == 1)[k][0]]
                        name.append(a)
                    name = list(set(name))
                    label_name.append(name)
        else:
            for j in range(0,np.shape(judge_array)[0]):
                temp = judge[i] - judge_array[j]
                if temp.any(): 
                    boolflag = boolflag
                else: #所有都为0，说明是已经留存过的组合
                    boolflag = 1
                    labels[i] = j+1 #由于顺序一致的兼容性，可以用j直接来代替
                    break
            if boolflag != 1: #这个表示一类
                labelstart = labelstart+1
                judge_array = np.vstack([judge_array,judge[i]])
                labels[i] = labelstart
                fig_labels.append(labelstart)
                
                name = [] #标签名自动计算
                for k in range(0,np.shape(np.argwhere(judge[i] == 1))[0]):
                    a = peaks_name[np.argwhere(judge[i] == 1)[k][0]]
                    name.append(a)
                name = list(set(name))
                label_name.append(name)
                
        if boolflag == 1:
            boolflag = 0
            continue
 
        # 峰位（2Theta）和峰强度数据
        peak_positions = np.array(df_peaks.iloc[i, 2:highs1_num])  
        peak_intensities = np.array(df_peaks.iloc[i, highs1_num:column_num])  
        peak_widths = np.zeros(angles_num)+0.1 # 用你的峰宽度数据替换
    
        # 绘制XRD谱
        x = np.linspace(min(peak_positions) - 5, max(peak_positions) + 5, 1000)  # 生成X轴坐标
        # 初始化XRD图谱
        xrd_spectrum = np.zeros_like(x)
    
        # 遍历每个峰并添加到XRD图谱
        for position, intensity, width in zip(peak_positions, peak_intensities, peak_widths):
            # 初始化参数值，可以根据实际数据进行调整
            initial = [intensity, position, width]
    
            # 将高斯峰添加到XRD图谱
            xrd_spectrum += gaussian(x, *initial)
        if boolflag == -1:
            xrd_spectra = xrd_spectrum
            boolflag = 0
        else:
            xrd_spectra = np.vstack([xrd_spectra, xrd_spectrum])
    #print(np.shape(judge_array))
    return xrd_spectra, peak_positions, peak_intensities, fig_labels, labels, label_name, judge_array

def plot_XRD_spectrum(xrd_spectra, peak_positions, peak_intensities, fig_labels,x_min = 20, x_max = 65): #绘制XRD不同分类结果
    titlename = "plot_XRD_spectrum"
    fig,ax=plt.subplots(figsize=(10,6))
    
    # 初始化一个基准强度，可以根据需要调整
    baseline_intensity = 1000
    
    for i, xrd_spectrum in enumerate(xrd_spectra):
        xrd_spectrum = xrd_spectrum / np.max(xrd_spectrum) * baseline_intensity
        x = np.linspace(min(peak_positions) - 5, max(peak_positions) + 5, 1000)  # 生成X轴坐标
        
        plt.plot(x, xrd_spectrum + i * baseline_intensity, label=fig_labels[i])
        
    # 添加图例和标签
    plt.xlabel('2Theta (°)')
    plt.ylabel('Intensity')
    plt.title('%s'%titlename)
    plt.grid(False)
    plt.legend(loc='best')
    
    # 调整y轴范围，确保不重叠
    plt.ylim(-500, len(xrd_spectra) * baseline_intensity + baseline_intensity)
    ax.axes.yaxis.set_ticks([]) #隐藏y轴ticks标签
    ax.set_xlim(xmin = x_min, xmax = x_max)
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith) #边框粗细
    
def generate_tri_cord(n=40):
    # 该子函数给出datalist每个数据点对应的空间坐标，plot scale取100，现用
    # 默认参数是最大行数==该行的点数。每行是等差数列。生成画三角形需要的坐标点（蛇形）
    x = []
    y = []
    for i in range(n,0,-1):
        width = (i - 1) / (n - 1)
        #if (i+1)%2: # 偶数行从右往左
        if (n-i)%2: # 偶数行从右往左(20230825改，否则40行会反过来，35行一致)
            for j in range(i):
                x.append(0.5 - (j - (i-1)/2) * width / i)
                y.append((n-i)/n * np.sqrt(3)/2)
        else: #奇数行从左往右
            for j in range(i):
                x.append(0.5 + (j - (i-1)/2) * width / i)
                y.append((n-i)/n * np.sqrt(3)/2)
    return x,y

def findelement3(setname): # 函数作用：找到setname里的元素成分
    if 'Te' in setname:
        X = 'Te'
    if 'Sn' in setname:
        X = 'Sn'
    if 'Se' in setname:
        X = 'Se'
    return X

def convert_cord2comp(sample_x,sample_y,element3 = 'Te'): # for GSX
    sqrt3 = np.sqrt(3)
    rho = {'Sb': 20, 'Te': 20, 'Ge': 20, 'Sn': 20, 'Se': 20}    #鍍膜縂厚度
    rel_mass = {'Sb': 121.75 / 6.697, 'Te': 127.6 / 6.25,
                'Ge': 72.61 / 5.35, 'Sn': 118.69 / 7.30, 'Se': 78.84 / 4.819}  # Ge72.61/5.35 Te127.6/6.25 Sb121.75/6.697   #原子量/密度
    baseline = {'Sb': 0, 'Te': 0, 'Ge': 0, 'Sn': 0, 'Se': 0}

    norm_x = (sample_x - np.min(sample_x)) / (np.max(sample_x) - np.min(sample_x))  # normalized to (0,1)
    norm_y = (sample_y - np.min(sample_y)) / (np.max(sample_x) - np.min(sample_x)) * sqrt3 / 2

    thickness = pd.DataFrame(columns=['Sb', element3, 'Ge'])
    thickness['Sb'] = 1 - norm_x - norm_y / sqrt3  # 樣品空間的百分比
    thickness[element3] = norm_x - norm_y / sqrt3
    thickness['Ge'] = 2 / sqrt3 * norm_y

    mol = pd.DataFrame(columns=['Sb', element3, 'Ge'])
    for elem in thickness:
        mol[elem] = (thickness[elem] * rho[elem] + baseline[elem]) / rel_mass[elem]
    mol_frac = pd.DataFrame(columns=['Sb', element3, 'Ge'])
    
    for elem in mol_frac:
        mol_frac[elem] = mol[elem] / mol.sum(axis=1)
        
    comp_x = mol_frac[element3] + 0.5 * mol_frac['Ge']
    comp_y = mol_frac['Ge'] * sqrt3 / 2
    
    return comp_x, comp_y

def plot_tri(ax,element3, labels, title = None, interpolation=False,terlinenum = 40,internal = 30, cmap = "jet", ordered_custom_order = []): #三元绘图
    x,y = generate_tri_cord(terlinenum)
    x,y = convert_cord2comp(x, y, element3)
    
    if element3 == 'Te':
        x = 100*x + 0.5*y
        y = 111*y
        
    if element3 == 'Sn':
        x = 100*x
        y = 112*y
        
    if element3 == 'Se':
        x = 100*x
        y = 112*y
    
    scale = 100
    ax.set_aspect(1)
    fig, tax = ternary.figure(ax=ax,scale=scale)
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
    #points = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
    #colors = plt.cm.rainbow(np.linspace(0, 1, 14))
    #print(labels)
    if labels.max()<13: #颜色逻辑，少于13用这套，大于13用jet cmap自动分颜色
        colors = [
        "#0077BB",  # Vivid Blue
        "#EE7733",  # Orange
        "#33B679",  # Bluish Green
        "#FFFF00",  # Yellow
        "#009CCC",  # Blue
        "#CC3311",  # Vermilion
        "#EE3377",  # Reddish Purple
        "#009988",  # Green
        "#33BBEE",  # Sky Blue
        "#FF0000",  # Red
        "#BBBBBB",  # Purple
        "#EE33EE",   # Magenta
        "#330066"
        ]
    else:
        colors = plt.cm.jet(np.linspace(0, 1, labels.max()+1))

    # 用于存储创建的散点对象和对应的标签
    scatter_objects = []
    sorted_labels = []
    countt = 0
    unique_labels = np.arange(0,labels.max()+1)
    print(unique_labels)
    for i in range(0,np.shape(labels)[0]):
        binn = np.argwhere(unique_labels == labels[i])
        unique_labels[np.where(unique_labels == labels[i])] = -1 #避免重复绘制label
        if binn.size > 0:
            scatter_obj = ax.scatter(x[i], y[i], marker='o', color=colors[labels[i]], label=f"Category {labels[i]}", picker=True)
            sorted_labels.append(f"Category {countt}")
            countt = countt+1
        else:
            scatter_obj = ax.scatter(x[i], y[i], marker='o', color=colors[labels[i]], picker=True)
        
        scatter_obj.set_gid(i)  # 使用set_gid设置唯一标识符
        scatter_objects.append(scatter_obj)
        
    # 获取当前图例的句柄和标签文本,ordered_custom_order额外给定信息，sorted_labels按顺序排列信息，unsorted_labels原信息
    handles, unsorted_labels = plt.gca().get_legend_handles_labels()
    # 假设这是我们想要的自定义顺序
    custom_order = sorted_labels
    # 根据自定义顺序创建新的handles和labels列表
    new_handles = [handles[unsorted_labels.index(lbl)] for lbl in custom_order]
    if ordered_custom_order:
        custom_order = ordered_custom_order
    new_labels = custom_order
    
    # 使用排序后的handles和labels创建图例
    ax.legend(new_handles, new_labels,loc='center left', bbox_to_anchor=(1, 0.5))
        
    plt.rc('font',family='Times New Roman')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Set ticks
    tax.ticks(axis='lbr', linewidth=1.5, multiple=20, fontsize = 16, offset=0.02)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.left_axis_label("Sb (%)", fontsize=18, offset=0.18)
    tax.right_axis_label("Ge (%)", fontsize=18, offset=0.18)
    tax.bottom_axis_label("%s"%element3 + " (%)", fontsize=18, offset=0.18)  # 
    # Set ticks
    tax.ticks(axis='lbr', linewidth=1.5, multiple=20, fontsize=16, offset=0.02)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.left_axis_label("Sb (%)", fontsize=18, offset=0.18)
    tax.right_axis_label("Ge (%)", fontsize=18, offset=0.18)
    tax.bottom_axis_label("%s" % element3 + " (%)", fontsize=18, offset=0.18)  #
    plt.title("%s\n"%(title), fontsize=16, fontweight='bold') #标题，2023/9/1隐去
    plt.subplots_adjust(top=0.85) #移动标题位置，当然其实可以tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize, offset=0.18)
    return fig, tax

def plot_tri_con(ax,element3, labels, title = None, interpolation=False,terlinenum = 40,internal = 30, cmap = "jet"): #三元连续图，暂时没有很成功，linear插值不太好用
    x,y = generate_tri_cord(terlinenum)
    x,y = convert_cord2comp(x, y, element3)
    
    if element3 == 'Te':
        x = 100*x + 0.5*y
        y = 111*y
        
    if element3 == 'Sn':
        x = 100*x
        y = 112*y
        
    if element3 == 'Se':
        x = 100*x
        y = 112*y
        
    scale = 100
    ax.set_aspect(1)
    fig, tax = ternary.figure(ax=ax,scale=scale)
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
    # 为了插值，创建一个网格
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 使用griddata进行插值
    zi = griddata((x, y), labels, (xi, yi), method='linear')

    # 绘制等高线图
    plt.contourf(xi, yi, zi, levels=np.arange(labels.min(), labels.max()+1), cmap='viridis', alpha=0.7)
    colors = plt.cm.jet(np.linspace(0, 1, 14))
    
    unique_labels = sorted(set(labels))  # 获取唯一的标签并排序 
    plt.rc('font',family='Times New Roman')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Set ticks
    tax.ticks(axis='lbr', linewidth=1.5, multiple=20, fontsize = 16, offset=0.02)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.left_axis_label("Sb (%)", fontsize=18, offset=0.18)
    tax.right_axis_label("Ge (%)", fontsize=18, offset=0.18)
    tax.bottom_axis_label("%s"%element3 + " (%)", fontsize=18, offset=0.18)  # 
    # Set ticks
    tax.ticks(axis='lbr', linewidth=1.5, multiple=20, fontsize=16, offset=0.02)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.left_axis_label("Sb (%)", fontsize=18, offset=0.18)
    tax.right_axis_label("Ge (%)", fontsize=18, offset=0.18)
    tax.bottom_axis_label("%s" % element3 + " (%)", fontsize=18, offset=0.18)  #
    plt.title("%s\n"%(title), fontsize=16, fontweight='bold') #标题，2023/9/1隐去
    plt.subplots_adjust(top=0.85) #移动标题位置，当然其实可以tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize, offset=0.18)
    return fig, tax

def plot_quin(arraydata, labels, title, ordered_custom_order = []):#五元绘图
    
    arraydata = np.array(arraydata)
    x = arraydata[:,0]
    y = arraydata[:,1]
    labels = np.array(labels)
    if labels.max()<13: #颜色逻辑，少于13用这套，大于13用jet cmap自动分颜色
        colors = [
        "#0077BB",  # Vivid Blue
        "#EE7733",  # Orange
        "#33B679",  # Bluish Green
        "#FFFF00",  # Yellow
        "#009CCC",  # Blue
        "#CC3311",  # Vermilion
        "#EE3377",  # Reddish Purple
        "#009988",  # Green
        "#33BBEE",  # Sky Blue
        "#FF0000",  # Red
        "#BBBBBB",  # Purple
        "#EE33EE"   # Magenta
        ]
    else:
        colors = plt.cm.jet(np.linspace(0, 1, labels.max()+1))
        
    fig, ax = plt.subplots(figsize = (10, 6))
    # 用于存储创建的散点对象和对应的标签
    scatter_objects = []
    sorted_labels = []
    countt = 0
    unique_labels = np.arange(0,labels.max()+1)
    print(unique_labels)
    for i in range(0,np.shape(labels)[0]):
        binn = np.argwhere(unique_labels == labels[i])
        unique_labels[np.where(unique_labels == labels[i])] = -1 #避免重复绘制label
        if binn.size > 0:
            scatter_obj = ax.scatter(x[i], y[i], marker='o', color=colors[labels[i]], label=f"Category {labels[i]}", picker=True)
            sorted_labels.append(f"Category {countt}")
            countt = countt+1
        else:
            scatter_obj = ax.scatter(x[i], y[i], marker='o', color=colors[labels[i]], picker=True)
        
        scatter_obj.set_gid(i)  # 使用set_gid设置唯一标识符
        scatter_objects.append(scatter_obj)
        
    # 获取当前图例的句柄和标签文本,ordered_custom_order额外给定信息，sorted_labels按顺序排列信息，unsorted_labels原信息
    handles, unsorted_labels = plt.gca().get_legend_handles_labels()
    # 假设这是我们想要的自定义顺序

    custom_order = sorted_labels
    # 根据自定义顺序创建新的handles和labels列表
    new_handles = [handles[unsorted_labels.index(lbl)] for lbl in custom_order]
    if ordered_custom_order:
        custom_order = ordered_custom_order
    new_labels = custom_order
    
    # 设置图例的字体和加粗
    legend_font = {'family': 'Times New Roman',   # 字体类型
                   'weight': 'bold',     } # 字体粗细         # 字体大小 'size': 10
    
    # 使用排序后的handles和labels创建图例
    ax.legend(new_handles, new_labels,loc='center left', bbox_to_anchor=(0.8, 0.5),prop=legend_font) 
    plt.title("%s\n"%(title), fontsize=16, fontweight='bold') #标题，2023/9/1隐去
    plt.subplots_adjust(top=0.85) #移动标题位置，当然其实可以tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize, offset=0.18)
    plt.axis('equal')
    plt.axis('off')
    return fig

def exchange(labels, former, latter=1, others='True'): #自定义交换顺序相关
    labels = np.array(labels)
    fa = np.arange(1, labels.max()+1, 1)
    
    # 包容了numpy数组和单数字，以及是否二分（others = 'False'表示二分）的所有情况，建议不要轻易修改逻辑
    if others =='True':
        if np.shape(former):
            for i in range(0,np.shape(former)[0]):
                labels[np.where(labels == former[i])[0]] = latter
        else:
            labels[np.where(labels == former)[0]] = latter
    else:
        if np.shape(former):
            cs = [item for item in fa if item not in former]
            for i in range(0,np.shape(cs)[0]):
                labels[np.where(labels == cs[i])[0]] = 0
            for i in range(0,np.shape(former)[0]):
                labels[np.where(labels == former[i])[0]] = 1 
        else:
            cs = [x for x in fa if x != former]
            for i in range(0,np.shape(cs)[0]):
                labels[np.where(labels == cs[i])[0]] = 0
            labels[np.where(labels == former)[0]] = 1
            
    return labels

def exchange2(labels, former, rename, latter=1, others='True'): #重命名绑定相关，自动添加了amorphous的名称，对应标签为0
    labels = np.array(labels)
    fa = np.arange(1, labels.max()+1, 1)
    
    # 包容了numpy数组和单数字，以及是否二分（others = 'False'表示二分）的所有情况，建议不要轻易修改逻辑
    if others =='True':
        if np.shape(former):
            for i in range(0,np.shape(former)[0]):
                labels[np.where(labels == former[i])[0]] = latter
                for j in range(former[i], labels.max()+1):
                    labels[np.where(labels == j+1)[0]] = j
                del rename[former[i]]
        else:
            if former != latter:
                item = np.where(labels == former)[0]
                labels[item] = latter
                for i in range(former, labels.max()+1):
                    labels[np.where(labels == i+1)[0]] = i
                del rename[former]
    else:
        rerename = []
        if np.shape(former):
            cs = [item for item in fa if item not in former]
            for i in range(0,np.shape(cs)[0]):
                labels[np.where(labels == cs[i])[0]] = 0
            for i in range(0,np.shape(former)[0]):
                labels[np.where(labels == former[i])[0]] = 1 
            rerename.append('amorphous') 
            rerename.append(str(rename[former[0]])+'only first peak') 
        else:
            cs = [x for x in fa if x != former]
            for i in range(0,np.shape(cs)[0]):
                labels[np.where(labels == cs[i])[0]] = 0
            labels[np.where(labels == former)[0]] = 1
            rerename.append('amorphous') 
            rerename.append(str(rename[former])) 
        rename = rerename            
    return labels, rename

def auto_exchange(labels, name): #自动化换名换标签
    i = 0
    countt = 0
    print(name)
    new_name = []
    added_name = []
    added_name.append('amorphous')
    
    #目前这里的逻辑是分为0-i和i+1-n的两个循环，第一个循环判断i是否重复，若没有重复，则将其作为1类并找到第二个循环里的相同类别，均作为这一类别，配合exchange函数完成
    for i in range(0,len(name)):
        boolflag = 0
        for k in range(0, i+1):
            if (set(name[k]) == set(name[i])) & (i!=k):
                boolflag = 1
        if boolflag == 1:
            continue
        else:
            countt = countt + 1
            new_name.append(name[i])
            labels = exchange(labels, i+1, latter=countt)
        for j in range(i+1, len(name)):
            if set(name[j]) == set(name[i]):
                labels = exchange(labels, j+1, latter=countt)
    for i in range(0,len(new_name)):
        added = new_name[i][0]
        for j in range(1,len(new_name[i])):
            added = added + '+' + new_name[i][j] 
        added_name.append(added)
    return labels, added_name

def every_peak(save_ori_labels, judge_array, peaks_name, printed_peak): #找到某个索引的峰
    rename = np.arange(0, save_ori_labels.max())
    a = np.where(judge_array[:,printed_peak-1] == 1)
    a = np.array(a)[0] + 1
    labels, rename = exchange2(save_ori_labels, a, rename, others='False')
    return labels, rename

def on_pick(event, directory, angle_center, r, sigma, window_size): #图形交互
    # 获取被点击对象的gid，即设置的迭代次数
    ind = event.artist.get_gid()
    # 获取对应的谱图数据
    '''
    str_ind = str(ind)
    n = int(3-len(str_ind))
    padded_string = add_zeros(str_ind, n)
    fig, ax = plt.subplots()
    for i in range(0,len(directory)):
        name = '%s-'%str(angle_center[i])+'%s'%padded_string+'-0000_exported'
        data = np.loadtxt(directory[i] + name + '.xy', encoding='utf-8', skiprows=1)
        x = data[:, 0]  # 2θ角度
        y = data[:, 1]  # 强度值
        y_ori = data[:, 1]  # 初始强度值
        y = preprocess_data(x, y, r=r, sigma=sigma, window_size=window_size)
        plt.plot(x, y)#去背底
        plt.plot(x, y_ori)#原始数据
        ax.set_title('%s'%name)
    fig.show()
    '''
    from pyxrd import XrdProcess
    str_ind = str(ind)
    n = int(3-len(str_ind))
    padded_string = add_zeros(str_ind, n)
    left_path = directory[0] + '%s-'%str(angle_center[0])+'%s'%padded_string+'-0000_exported' + '.xy'
    right_path = directory[1] + '%s-'%str(angle_center[1])+'%s'%padded_string+'-0000_exported' + '.xy'
    # Process data
    data = XrdProcess(
        file_path=[left_path, right_path],
        pattern_path='data/GeSbSn_icsd/pattern.pkl',
        structure_path='data/GeSbSn_icsd/structure.pkl',
    )
    data.identify(
        figure_title=f'100 nm-300C-1h-{ind}', 
        display_number=8,
        tolerance=0.2,
    )

def add_zeros(string, n): #自动填充零
    # 使用 zfill() 方法在字符串的左侧填充零
    padded_string = string.zfill(len(string) + n)
    return padded_string

def plot_merge_XRD(directory,name,angles,r=5,sigma=20,window_size=50,updown_gap= 2):#逻辑，如果下一个角和上一个角不重合，则无需计算交叉部分，否则，交叉部分取前一个角度的信息，正负13度作为参考，每个角度扣除前1度的信息
    # 遍历数据列，绘制XRD谱图
    #计算过程1
    fig,ax=plt.subplots(figsize=(10,6))
    
    # 为不同列设置颜色，你可以根据需要自定义颜色
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    for i in range(0,len(name[0])):
        x_merge = []
        y_merge = []
        original_deduction = 13
        for j in range(0,len(name)):
            data = np.loadtxt(directory[j] + name[j][i] + '.xy', encoding='utf-8', skiprows=1)
            x = data[:, 0]  # 2θ角度
            y = data[:, 1]  # 强度值
            cut = np.where((x>angles[j]-original_deduction) &(x<angles[j]+13))
            x = x[cut]
            y = y[cut]
            if j == 0:
                x_merge = x
                y_merge = y
            else:
                x_merge = np.append(x_merge, x)
                y_merge = np.append(y_merge, y)
            if j == len(name) - 1:
                original_deduction = 13    
            else:
                if angles[j+1] - angles[j] <28:
                    original_deduction = angles[j+1] - angles[j]- 13  
        y_merge = preprocess_data(x_merge, y_merge, r=r, sigma=sigma, window_size=window_size)
        plt.plot(x_merge, y_merge + updown_gap*i, label='(%s)'%name[0][i], color=colors[i % len(colors)])
        plt.show()
        
    # 添加图例和标签
    plt.legend(loc='best')
    plt.xlabel('2Theta (°)')
    plt.ylabel('Intensity')
    plt.title('XRD Spectra')
    plt.grid(False)
    
    # 调整y轴范围，确保不重叠
    plt.ylim(0, 10)
    #ax.axes.yaxis.set_ticks([]) #隐藏y轴ticks标签
    
    
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith) #边框粗细
    # 显示图形
    plt.show()
    
    #计算过程2
    fig,ax=plt.subplots(figsize=(10,6))
    
    # 为不同列设置颜色，你可以根据需要自定义颜色
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    # 遍历数据列，绘制XRD谱图
    for i in range(0,len(name[0])):
        x_merge = []
        y_merge = []
        original_deduction = 13
        for j in range(0,len(name)):
            data = np.loadtxt(directory[j] + name[j][i] + '.xy', encoding='utf-8', skiprows=1)
            x = data[:, 0]  # 2θ角度
            y = data[:, 1]  # 强度值
            cut = np.where((x>angles[j]-original_deduction) &(x<angles[j]+13))
            x = x[cut]
            y = y[cut]
            #print(y)#y = preprocess_data(x, y, r=r, sigma=sigma, window_size=window_size)
            #print(x)
            #print(y)
            if j == 0:
                x_merge = x
                y_merge = y
            else:
                x_merge = np.append(x_merge, x)
                y_merge = np.append(y_merge, y)
            if j == len(name) - 1:
                original_deduction = 13    
            else:
                if angles[j+1] - angles[j] <28:
                    original_deduction = angles[j+1] - angles[j]- 13 
        
        plt.plot(x_merge, y_merge + updown_gap*i, label='(%s)'%name[0][i], color=colors[i % len(colors)])
        plt.show()
        
    # 添加图例和标签
    plt.legend(loc='best')
    plt.xlabel('2Theta (°)')
    plt.ylabel('Intensity')
    plt.title('XRD Spectra')
    plt.grid(False)
    
    # 调整y轴范围，确保不重叠
    plt.ylim(0, 10)
    #ax.axes.yaxis.set_ticks([]) #隐藏y轴ticks标签
    
    
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith) #边框粗细
    # 显示图形
    plt.show()

def exchange3(labels, point_Num, rename, latter=1, others='True'): #重命名绑定相关（改变某个点），如果False二分类自动添加了amorphous的名称，对应标签为0
    labels = np.array(labels)
    point_Num = np.array(point_Num)
    # 包容了numpy数组和单数字，以及是否二分（others = 'False'表示二分）的所有情况，建议不要轻易修改逻辑
    
    if not np.shape(point_Num): #point_Num是单个数字
        if labels[point_Num] == latter: #避免重复性替代
            return labels, rename
        if np.count_nonzero(labels == labels[point_Num]) != 1:
            if others =='True':
                labels[point_Num] = latter
            else:
                rerename = []
                labels = np.zeros_like(labels)
                labels[point_Num] = 1
                rerename.append('amorphous') 
                rerename.append(str(point_Num)) 
                rename = rerename            
        else:
            labels, rename = exchange2(labels, labels[point_Num], rename, latter, others)
    else: #point_Num是个list
        point_labels = labels[point_Num]
        uni_point_labels = np.sort(np.unique(point_labels))
        if others =='True':
            for i in range(0, np.shape(uni_point_labels)[0]):
                item_label = uni_point_labels[i]
                if item_label == latter:#避免重复性替代
                    continue
                if np.count_nonzero(labels == item_label) > np.count_nonzero(point_labels == item_label):
                    labels[point_Num[np.argwhere(point_labels == item_label)]] = latter
                else:
                    labels, rename = exchange2(labels, item_label, rename, latter, others)
                    point_labels = point_labels - 1
                    uni_point_labels = uni_point_labels - 1 
        else:
            rerename = []
            labels = np.zeros_like(labels)
            labels[point_Num] = 1
            rerename.append('amorphous') 
            rerename.append(str(point_Num)) 
            rename = rerename             
        
    return labels, rename