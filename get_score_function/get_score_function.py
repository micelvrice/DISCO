import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Polynomial
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
dataset_list = ["narrativeqa", "qasper", "multifieldqa_en",  "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

model = "Mistral-7b-Instruct-v0.3-32k"
all_key_arrays = []
all_value_arrays = []
assert len(dataset_list) == 16, f"Expected 16 datasets, but got {len(dataset_list)}"

for dataset in dataset_list:
    # 在25服务器/home/sjx/papers/cakekv/experiments/LongBench/kv_cache_sparse
    path_key = f"kv_cache_sparse/{model}/pca_{4}/key_{dataset}.pt"
    path_value = f"kv_cache_sparse/{model}/pca_{4}/value_{dataset}.pt"
    key_results = torch.load(path_key)
    value_results = torch.load(path_value)
    key_array = np.array(key_results)
    value_array = np.array(value_results)
    all_key_arrays.append(key_array)
    all_value_arrays.append(value_array)

assert len(all_value_arrays) == 16
assert all_key_arrays[0].shape == (20, 32)

combined_key_array = np.concatenate(all_key_arrays, axis=0)
combined_value_array = np.concatenate(all_value_arrays, axis=0)

from scipy.ndimage import gaussian_filter1d

def smooth_gaussian(data, sigma=1.0):
    """
    对每个样本的32维特征使用一维高斯滤波
    sigma 控制平滑程度
    """
    smoothed = np.copy(data)
    for i in range(data.shape[0]):
        smoothed[i] = gaussian_filter1d(data[i], sigma=sigma)
    return smoothed

from scipy.interpolate import UnivariateSpline

def smooth_spline(data, s=1.0):
    smoothed = np.copy(data)
    x = np.arange(32)
    for i in range(data.shape[0]):
        spline = UnivariateSpline(x, data[i], s=s)
        smoothed[i] = spline(x)
    return smoothed


def minmax_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


norm_value = minmax_normalize(combined_value_array)
norm_key = minmax_normalize(combined_key_array)

norm = norm_key + norm_value

assert norm.shape == (320, 32)

avg_per_layer = norm.mean(axis=0)
poly_fit = Polynomial.fit(x, avg_per_layer, deg=6)

np.save("full_fit_mistral_combined16_pca4.npy", poly_fit.convert().coef)