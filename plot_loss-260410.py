import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from torch.utils.data import DataLoader
import torch
from paras12 import *
from algorithm_toe import *

data_root   = "res/data_8td_randA_randpphi_snr_train_k1_2/"
file_path_lista = data_root + "LISTA_epoch_9_log_250319"
df_lista = pd.read_csv(file_path_lista)
# file_path_clista = data_root + "ComplexLISTA_epoch_9_log_250319"
# df_clista = pd.read_csv(file_path_clista)
file_path_lista_toe = data_root + "Toe_LISTA_epoch_9_log_250319"
lista_toe = pd.read_csv(file_path_lista_toe)
file_path_ad_alista_toe = data_root + "Toe_LISTA_Ada_epoch_9_log_250319"
ad_alista_toe = pd.read_csv(file_path_ad_alista_toe)
plt.rcParams['font.size'] = 16  # 设置全局字体大小
plt.rcParams['font.family']=' Times New Roman, SimSun'# 设置字体族，中文为SimSun，英文为Times New Roman
# plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
# 定义颜色映射
COLOR_MAP = {
    'LISTA': [0.02, 0.85, 0.85],
    # 'CLISTA': [0.32, 0.55, 0.85],
    # 'gamma_net': [0.5,0.5,0.58],
    # 'CLISTA-Ada': [0.99,0.43,0.35],
    'Toe-LISTA': [0.27,0.2,0.58],
    'Toe-CLISTA-Ada': [1,0.71,0.04]
}

def plot_loss(x, data_dict, title, ylabel, save_name, fontsizeset=20):
    """
    绘制损失曲线并保存图像
    :param x: x 轴数据
    :param data_dict: 包含标签和数据的字典，例如 {'label': data}
    :param title: 图表标题
    :param ylabel: y 轴标签
    :param save_name: 保存图像的文件名
    :param fontsizeset: 字体大小
    """
    plt.figure(figsize=(10, 6))  # 设置图表大小
    for label, data in data_dict.items():
        plt.plot(x, data, linewidth=3, label=label, color=COLOR_MAP.get(label, 'black'))  # 绘制曲线，设置线条宽度为 2
    
    plt.xlabel('训练轮数', fontsize=fontsizeset)  # 设置 x 轴标签
    plt.ylabel(ylabel, fontsize=fontsizeset)  # 设置 y 轴标签
    plt.title(title, fontsize=fontsizeset)  # 设置图表标题
    # plt.legend(fontsize=fontsizeset)  # 显示图例，设置字体大小

    
    # 设置图例
    legend = plt.legend(
        fontsize=fontsizeset,  # 设置字体大小
        borderpad=0.2,  # 减少边框内部填充
        labelspacing=0.2,  # 减少条目之间的垂直间距
        handlelength=1.5,  # 缩短线条/标记的长度
        handletextpad=0.2,  # 减少线条/标记与文本之间的间距
        borderaxespad=0.2,  # 减少图例与坐标轴边框之间的间距
        frameon=False,  # 去掉图例边框
        ncol=1  # 将图例分为两列
    )
    # 调整布局并显示
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 调整布局
    plt.savefig(save_name, dpi=300, bbox_inches='tight')  # 保存图像
    plt.show()
    plt.close()  # 关闭当前图表
# 定义 x 轴数据
x = np.arange(len(df_lista['test_loss'])) / 300  # 将索引映射到 epoch

# 绘制测试损失曲线
test_loss_data = {
    'LISTA': df_lista['test_loss'],
    # 'CLISTA': df_clista['test_loss'],
    # 'gamma_net': df_gammanet['test_loss'],
    # 'CLISTA-Ada': lista_soft['test_loss'],
    'Toe-LISTA': lista_toe['test_loss'],
    'Toe-CLISTA-Ada': ad_alista_toe['test_loss']
}

plot_loss(x, test_loss_data, '', 'test loss', "saunfa_test_loss_snr.jpg")
# 构造要保存的字典

# 保存为 .mat 文件
io.savemat(os.path.join(data_root, 'loss.mat'), test_loss_data)