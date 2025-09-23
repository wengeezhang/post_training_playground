from datasets import load_dataset,DownloadConfig
from torch.utils.data import DataLoader

print("Loading dataset...")
dataset = load_dataset("csv", data_files="prompts.csv", split="train")


# 打印数据集基本信息
print(f"Dataset size: {len(dataset)} examples")
print(f"Dataset features: {dataset.features}")

dataloader_params = {
            "batch_size": 10,
            "num_workers": 1,
        }

dataloader = DataLoader(dataset, **dataloader_params)

print(f"dataloader size: {len(dataloader)}")