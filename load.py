import pandas as pd
import torch
# ssss = pd.read_csv('./dataset/traffic.csv')
import json
import numpy as np
# # 设置显示的最大列数为None，以便展示所有列
# pd.set_option('display.max_columns', None)
#
# # 读取CSV文件的前100行
# df = pd.read_csv('./dataset/traffic.csv', nrows=100)
#
# # 显示结果
# print(df)

file_path = './dataset/traffic.csv'  # 将 'your_file_path.cat' 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    for i in range(100):
        line = file.readline()
        if not line:
            break  # 如果文件不足100行，提前结束循环


# 创建一个二维张量
a = torch.tensor([[1, 2], [3, 4]])

# 沿着第一个维度（dim=0）重复每个元素2次
b = torch.repeat_interleave(a, 2, dim=1)
print(b)
