import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import warnings
import argparse
import random
import json
import os
model_path = {
    "Llama3-8b-Instruct-8k": "/home/sjx/kv/pre_trained_models/Llama-3-8B-Instruct-8k",
    "llama3.1-8b-128k": "/home/sjx/kv/pre_trained_models/Llama-3.1-8B-Instruct",
    "Mistral-7b-Instruct-v0.3-32k": "/home/sjx/kv/pre_trained_models/Mistral-7b-Instruct-v0.3-32k",
    "ERNIE-4.5-21B-A3B-PT": "/home/sjx/pretrained_models/ERNIE-4.5-21B-A3B-PT"
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ERNIE-4.5-21B-A3B-PT", 
                        choices=["Llama3-8b-Instruct-8k",  
                                 "llama3.1-8b-128k", 
                                 "Mistral-7b-Instruct-v0.3-32k",
                                 "ERNIE-4.5-21B-A3B-PT"])
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--dataset', type=str, default="qasper")
    parser.add_argument('--pca_components', type=int, default=4)
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        print("======== llama3 build chat ========")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "ERNIE" in model_name:
        print("======== ERNIE build chat ========")
        prompt = f"<|begin_of_sentence|> {prompt} <|end_of_sentence|>"
    elif "mistral" in model_name:
        print("======== mistral build chat ========")
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        print("======== qwen build chat ========")
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


def calculate_sparse_v(past_key_values):
    results = []
    num_layers = len(past_key_values)
    for layer in range(num_layers):
        value = past_key_values[layer][1].squeeze(0)
        pca_n_components = 2
        seq_len = value.shape[1]
        value = value.permute(1,0,2).reshape(seq_len, -1)
        pca = PCA(n_components=pca_n_components)
        value_2d = pca.fit_transform(value.cpu().numpy())
        explained_variance = pca.explained_variance_
        dispersion_score = explained_variance.sum()
        results.append(dispersion_score)
    return results

def calculate_sparse_k(past_key_values):
    results = []
    num_layers = len(past_key_values)
    for layer in range(num_layers):
        key = past_key_values[layer][1].squeeze(0)
        pca_n_components = 2
        seq_len = key.shape[1]
        key = key.permute(1,0,2).reshape(seq_len, -1)
        pca = PCA(n_components=pca_n_components)
        key_2d = pca.fit_transform(key.cpu().numpy())
        explained_variance = pca.explained_variance_
        dispersion_score = explained_variance.sum()
        results.append(dispersion_score)
    return results

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_args()
    model_path = "/home/sjx/pretrained_models/ERNIE-4.5-21B-A3B-PT"

    device = torch.device(f"cuda:{args.device}")
    dtype=torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(device)
    model = model.eval()
    datasetmaxlen = 128
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    data = load_dataset('THUDM/LongBench', args.dataset, split='test')
    data_shuffled = data.shuffle(seed=42)
    data_all = data_shuffled.select(range(20))
    prompt_format = dataset2prompt[args.dataset]
    max_length = 127500

    all_past_cache = []
    key_results, value_results = [], []
    for idx, json_obj in enumerate(data_all):
        print(f"process------{idx+1}/20-----")
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if args.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, args.model)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        print(context_length)
        gen_kwargs = {"do_sample": True, "top_k": 1, "return_dict_in_generate": True}
        if args.dataset == "samsum":
            outputs = model.generate(
                **input,
                max_new_tokens=1,
                **gen_kwargs,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )
        else:
            outputs = model.generate(
                **input,
                **gen_kwargs,
                max_new_tokens=1
            )
            
        past_key_values = outputs['past_key_values']

        k_sparse = calculate_sparse_k(past_key_values)
        v_sparse = calculate_sparse_v(past_key_values)

        key_results.append(k_sparse)
        value_results.append(v_sparse)

    output_dir = f"kv_cache/{args.model}/pca_{args.pca_components}"
    os.makedirs(output_dir, exist_ok=True)
    # 保存 key 和 value 稀疏性结果
    torch.save(key_results, os.path.join(output_dir, f"key_{args.dataset}.pt"))
    torch.save(value_results, os.path.join(output_dir, f"value_{args.dataset}.pt"))

        

