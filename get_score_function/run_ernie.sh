#!/bin/bash

MODEL_NAME="ERNIE-4.5-21B-A3B-PT"

DEVICE=0
PCA_COMPONENTS=4

# 你要处理的所有数据集
datasets="narrativeqa multifieldqa_en qasper hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
for dataset in $datasets
do
  echo "Running dataset: $dataset with model: $MODEL_NAME"
  python get_layer_sparse.py \
    --model "$MODEL_NAME" \
    --device "$DEVICE" \
    --dataset "$dataset" \
    --pca_components "$PCA_COMPONENTS"
done