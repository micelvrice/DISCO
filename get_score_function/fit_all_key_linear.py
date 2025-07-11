import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Polynomial
import random
model_path = {
    "Llama3-8b-Instruct-8k": "/home/sjx/kv/pre_trained_models/Llama-3-8B-Instruct-8k",
    "Llama-3.1-8B-Instruct-128k": "/home/sjx/kv/pre_trained_models/Llama-3.1-8B-Instruct",
    "Mistral-7b-Instruct-v0.3-32k": "/home/sjx/kv/pre_trained_models/Mistral-7b-Instruct-v0.3-32k"
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k", 
                        choices=["llama3.1-8b-128k",  
                                 "Llama-3.1-8B-Instruct-128k", 
                                 "Mistral-7b-Instruct-v0.3-32k",])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pca_n_components', type=int, default=2)
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def minmax_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # 多个数据集的名称（你可以根据需要修改）
    dataset_list = ["narrativeqa", "qasper", "multifieldqa_en",  "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    all_key_arrays, all_value_arrays = [], []

    for dataset in dataset_list:
        path_key = f"kv_cache/{args.model}/pca_{args.pca_n_components}/key_{dataset}.pt"
        path_value = f"kv_cache/{args.model}/pca_{args.pca_n_components}/value_{dataset}.pt"
        key_results = torch.load(path_key)
        value_results = torch.load(path_value)
        key_array = np.array(key_results)
        value_array = np.array(value_results)
        all_key_arrays.append(key_array)
        all_value_arrays.append(value_array)

    combined_key_array = np.concatenate(all_key_arrays, axis=0)
    combined_value_array = np.concatenate(all_value_arrays, axis=0)

    norm_key = minmax_normalize(combined_key_array)
    norm_value = minmax_normalize(combined_value_array)

    norm = norm_key + norm_value
    # 每层的平均稀疏度
    avg_per_layer = norm.mean(axis=0)
    avg_per_layer_value = combined_value_array.mean(axis=0)
    x = np.arange(len(avg_per_layer))

    # 多项式拟合
    poly_fit = Polynomial.fit(x, avg_per_layer, deg=6)
    y_poly = poly_fit(x)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_per_layer, label='Avg Sparse Score (all datasets)', marker='o', linewidth=1)
    plt.plot(x, y_poly, label='Poly Fit (deg=6)', linestyle='--')
    plt.xlabel("Layer")
    plt.ylabel("Sparse Score")
    plt.title(f"Layer-wise Sparse Score Trend\n(Model: {args.model})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{args.model}_pca{args.pca_n_components}.png')

    # 保存标准多项式系数（可用于预测）
    poly_std = poly_fit.convert()
    np.save(f"score_function_{args.model}_pca{args.pca_n_components}.npy", poly_std.coef)

