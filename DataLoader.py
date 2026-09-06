import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import pickle


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cal_percentage(data, percentage):
    return torch.quantile(data, percentage)
    
def clip_data(data, threshold):
    return torch.where(data > threshold, threshold, data)

def normalize_cond(cond):
    # 计算 K 维度上的最小值和最大值，保持 N, L 维度
    min_vals = cond.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)  # shape (1, K, 1)
    max_vals = cond.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)  # shape (1, K, 1)

    # 避免除零错误（如果 min == max，则保持原值）
    normed = (cond - min_vals) / (max_vals - min_vals + 1e-8) * 2 - 1

    return normed

def normalize_MAG(MAG):
    """
    对形状 (N, 2, 64, 64) 的 MAG 的两个通道分别进行全局归一化
    每个通道使用该通道全体数据的 min/max
    
    返回：归一化后的 MAG (float32)
    """
    MAG = MAG.astype(np.float32)
    MAG_norm = MAG.copy()

    # 对每个通道分别归一化
    for c in range(2):
        channel = MAG[:, c, :, :] # (N, 64, 64)
        min_val = channel.min()
        max_val = channel.max()

        # 避免除 0
        if max_val > min_val:
            MAG_norm[:, c, :, :] = (channel - min_val) / (max_val - min_val)
        else:
            MAG_norm[:, c, :, :] = 0.0   # 通道全相等时直接设为 0

    return MAG_norm

def data_load_single(args, datatype):
    
    # 数据集加载 ————————————————————————————————————————————————————————————————————————————
    X, calc, C, MAG, TRAJ, ref_emb = raw_load(args, datatype) # N, L

    # 处理数据项 X ================================================
    X = X[:, :args.time_length] # N, L
    calc = calc[:, :args.time_length]
    print("数据集规模：", X.shape[0])
    print("数据最大值：", X.max())

    clip_max = np.percentile(X, 99)
    clip_min = np.percentile(X, 1)
    X = np.clip(X, a_min=clip_min, a_max=clip_max)

    clip_max = np.percentile(calc, 99)
    clip_min = np.percentile(calc, 1)
    calc = np.clip(calc, a_min=clip_min, a_max=clip_max)

    args.seq_len = X.shape[-1]

    if args.prompt_state == 'load':
        # 数据集划分
        path_idx = args.model_path + 'model_save/idx_' + datatype + '.pk'
        with open(path_idx, "rb") as f:
            train_idx, val_idx, test_idx = pickle.load(f)
        # 归一化
        path_scaler = args.model_path + 'model_save/scalers/scaler_' + datatype + '.pk'
        with open(path_scaler, "rb") as f:
            scaler = pickle.load(f)
    else:
        np.random.seed(None)
        # 首先划分训练集和临时集（验证集 + 测试集）

        if datatype == 'RSRP-CPGMCM':
            train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=24)
        else:
            train_idx = np.arange(int(len(X)*0.8))
            np.random.shuffle(train_idx)
            test_idx = np.arange(int(len(X)*0.8), len(X))
        
        if args.prompt_state in ['zs', 'fs']:
            train_size = 0.05 if args.prompt_state == 'zs' else args.fs_rate
            train_idx, _ = train_test_split(np.arange(len(train_idx)), train_size=train_size, random_state=24)
        
        val_idx = test_idx
        np.random.shuffle(val_idx)

        # 归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X[train_idx].reshape(-1,1))

    # 对所有子集进行标准化
    data_scaled = scaler.transform(X.reshape(-1,1)).reshape(X.shape)
    calc_scaled = scaler.transform(calc.reshape(-1,1)).reshape(calc.shape)
    data_scaled = np.clip(data_scaled, a_min=-1, a_max=1)

    # 处理条件数据 ====================================================
    cond = np.array(C[:, :, :args.time_length]) # (N, K, L)
    args.feature_size = cond.shape[1]

    # 处理多属性图 ====================================================
    if args.use_mag == False:
        if args.use_prior:
            data = [[data_scaled[i], calc_scaled[i], cond[i], X[i], ref_emb[i]] for i in range(X.shape[0])]
        else:
            data = [[data_scaled[i], calc_scaled[i], cond[i], X[i]] for i in range(X.shape[0])]

    else:
        mag = MAG
        traj = TRAJ[:, :args.time_length]
        ref_emb = ref_emb[:, :args.time_length]
        if args.use_BH_map == False and args.use_AL_map == True:
            mag = np.expand_dims(mag[:, 0], axis=1)
        if args.use_BH_map == True and args.use_AL_map == False:
            mag = np.expand_dims(mag[:, 1], axis=1)
            
        args.mag_dim = mag.shape[2]
        args.mag_size = mag.shape[-1]
        data = [[data_scaled[i], calc_scaled[i], cond[i], X[i], mag[i], traj[i], ref_emb[i]] for i in range(X.shape[0])]

    # 创建 real 数据集 ————————————————————————————————————————————————————————————————————————————
    dataset = MyDataset(data)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 保存 scaler 和 indices
    with open(args.model_path + 'model_save/scaler_' + datatype + '.pk', "wb") as f:
        pickle.dump(scaler, f)
    with open(args.model_path + 'model_save/idx_' + datatype + '.pk', "wb") as f:
        pickle.dump([train_idx, val_idx, test_idx], f)

    return train_loader, test_loader, val_loader, scaler

def data_load_mix(args, data_list):
    data_all = []

    for data in data_list:
        data_all += data

    data_all = th.utils.data.DataLoader(data_all, batch_size=args.batch_size, shuffle=True)

    return data_all

def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('_'):
        data, test_data, val_data, my_scaler = data_load_single(args, dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append([dataset_name, test_data])
        val_data_all.append([dataset_name, val_data])
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name, i) for name, data in data_all for i in data]
    test_data_all = [(name, i) for name, test_data in test_data_all for i in test_data]
    val_data_all = [(name, i) for name, val_data in val_data_all for i in val_data]
    
    return data_all, test_data_all, val_data_all, my_scaler_all

def clean_tensor(x, fill_value=-50):
    # 替换 nan 和 inf
    x = torch.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return x

# 处理 numpy array 的情况
def clean_numpy(x, fill_value=0):
    x = np.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return x

def raw_load(args, datatype):
    
    # 数据读取
    path = '../datasets/' + datatype + '.npz'
    file = np.load(path, allow_pickle=True)
    data = np.array(file['data'], dtype=np.float32)
    calc = np.array(file['calc'], dtype=np.float32)
    # cond = np.load("../datasets/VAE_encoded_datasets/encoded_" + datatype + ".npy", allow_pickle=True)
    cond = np.array(file['cond'], dtype=np.float32) if args.ray_condition == 'OF' else np.array(file['cond_Nb'], dtype=np.float32)
    MAG = file['mag'] if 'mag' in file.files else None
    TRAJ = file['traj'] if 'traj' in file.files else None
    ref_emb = file['ref_emb'] if 'ref_emb' in file.files else None

    cond[:, 0, :] = 0

    args.use_mag = args.use_mag if 'mag' in file.files and args.prompt_state not in ['zs', 'fs'] else False
    
    # 格式转换
    data = torch.tensor(data, dtype=torch.float32) # (N, L)
    calc = torch.tensor(calc, dtype=torch.float32).reshape(data.shape) # (N, L)
    cond = np.array(cond, dtype=np.float32) # (N, K, L)
    cond = normalize_cond(cond)

    if args.use_mag:
        MAG = np.array(MAG, dtype=np.float32) # (N, 2, 64, 64)
        TRAJ = np.array(TRAJ, dtype=np.float32) # (N, L, 2)

        MAG = np.clip(MAG, a_min=0, a_max=np.percentile(MAG, 99))
        MAG = normalize_MAG(MAG)
    
    # 排除 nan 值
    data = clean_tensor(data, fill_value=-50)
    calc = clean_tensor(calc, fill_value=-50)
    cond = clean_numpy(cond, fill_value=0)
    MAG  = clean_numpy(MAG,  fill_value=0)

    # print(np.isnan(MAG).any())
    # print(np.isinf(MAG).any())

    # 定量输出
    data_num = 4000 if datatype == 'RSRP-mix' else 2000
    if len(data) > data_num:
        if args.use_mag:
            return data[:data_num], calc[:data_num], cond[:data_num], MAG[:data_num], TRAJ[:data_num], ref_emb[:data_num]
        elif ref_emb != None:
            return data[:data_num], calc[:data_num], cond[:data_num], None, None, ref_emb[:data_num]
        else:
            return data[:data_num], calc[:data_num], cond[:data_num], None, None, None
    else:
        if args.use_mag:
            return data, calc, cond, MAG, TRAJ, ref_emb
        elif ref_emb != None:
            return data, calc, cond, None, None, ref_emb
        else:
            return data, calc, cond, None, None, None