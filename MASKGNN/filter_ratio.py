import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 文件夹路径
# folder_path = r'E:/Academicfiles/pycharmfiles/Key-AST SC/data/SmartBugs/2vul_above_0_keynodes'
folder_path = r'E:/Academicfiles/pycharmfiles/Key-AST SC/data/key_v_above_0_keynodes'

# 初始化比值列表
ratios = []

# 遍历文件夹中的所有.pt文件
for filename in os.listdir(folder_path):
    if filename.endswith(".pt"):
        file_path = os.path.join(folder_path, filename)
        data = torch.load(file_path)

        original_node_count = data['original_node_count']
        filtered_node_count = data['filtered_node_count']

        # 计算比值并添加到列表中
        if original_node_count != 0:  # 防止除以0
            ratio = filtered_node_count / original_node_count
            ratios.append(ratio)

with open('data/BJUT_ratios.pkl', 'wb') as file:
    pickle.dump(ratios, file)

# 定义比值范围
bins = np.arange(0, 1.1, 0.1)

# 计算每个范围内的数量
hist, bin_edges = np.histogram(ratios, bins=bins)

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.1, edgecolor='black', align='edge')

# 添加标签和标题
plt.xlabel('Ratio of filtered_node_count/original_node_count')
plt.ylabel('Count')
plt.title('Distribution of filtered_node_count/original_node_count Ratios')
plt.xticks(bin_edges)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.show()