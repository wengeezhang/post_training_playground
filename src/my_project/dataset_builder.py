import os

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader



print("Loading dataset... in dl")

datasetBuilder = load_dataset_builder("csv", data_files="prompts.csv", split="train")

# 打印数据集基本信息
print("Loaded dataset... in dl")

print(f"datasetBuilder features: {datasetBuilder.info.features}")

