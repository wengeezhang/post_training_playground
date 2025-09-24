import time

from datasets import load_dataset,DownloadConfig
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
# set start time
startTime = time.time()
print("Loading dataset... in init")


dataset = load_dataset("csv", data_files="prompts.csv", split="train")

print(f"Loaded time: {time.time() - startTime} seconds")

# 打印数据集基本信息
print(f"Dataset size: {len(dataset)} examples")
print(f"Dataset features: {dataset.features}")

# 打印第一个样例
print("\nFirst example:")
first_example = dataset[0]
for key, value in first_example.items():
    print(f"{key}: {value}")

# print first sample of this dataset

# print(dataset[0])

# define a var and print it
# model_id = "post_training_playground"
#
# print(model_id)