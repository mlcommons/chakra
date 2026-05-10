#!/bin/bash
tar -xzf chakra_memory_extracts.tar.gz

cd "$(dirname "$0")"

python plot_memory.py \
    chakra_memory_DeepSeek-MoE-8GPUs.csv \
    chakra_memory_GPT3-175B-32GPUs.csv \
    chakra_memory_GPT3-5B-8GPUs.csv \
    chakra_memory_Llama3-70B-16GPUs.csv \
    chakra_memory_Mixtral-8x22B-32GPUs.csv \
    chakra_memory_Mixtral-8x7B-8GPUs.csv \
    memory_all_models.pdf
