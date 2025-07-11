import json
import os
import argparse
import numpy as np
# 解析命令行参数
parser = argparse.ArgumentParser(description="Reorder a JSON file according to a predefined key sequence.")
parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
args = parser.parse_args()
input_path = args.input

# 读取 JSON 数据
with open(input_path, 'r') as f:
    data = json.load(f)

# 指定新的顺序
order = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

# 按顺序重新排列
reordered_data = {key: data[key] for key in order if key in data}

# 输出路径：同目录
output_path = os.path.join(os.path.dirname(input_path), "reordered_results.json")
with open(output_path, "w") as f:
    json.dump(reordered_data, f, indent=4)

print(f"Reordered data saved to: {output_path}")

# 打印每个任务对应的 value 值，格式化为 XX & XX & ...
# 拼接所有 value 值到一个字符串中
all_values = []
for value in reordered_data.values():
    if isinstance(value, list):
        all_values.extend(value)
    else:
        all_values.append(value)

# 格式化为一行：XX & XX & XX & ...
formatted_line = ' & '.join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in all_values) + ' &'
mean_value = np.mean(all_values)

print("\nFormatted values:")
print(formatted_line)
print(f"\nMean value: {mean_value:.2f}")