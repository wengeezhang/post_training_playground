import os

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


print("Loading dataset... in dl")

dataset = load_dataset("csv", data_files="prompts.csv", split="train")
dataset.set_format(type='torch')
# 打印数据集基本信息
print(f"Dataset size: {len(dataset)} examples")
print(f"Dataset features: {dataset.features}")

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    # 将prompt字段tokenize
    return tokenizer(
        examples["prompt"],
        padding=True,
        truncation=True,
        max_length=512
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'act', 'prompt'])


dataloader_params = {
    "batch_size": 10,
    "num_workers": 0,
}

dataloader = DataLoader(tokenized_dataset, **dataloader_params)

print(f"dataloader size: {len(dataloader)}")
print(f"dataloader.dataset size: {len(dataloader.dataset)}")

for batch in dataloader:
    tokens = batch["input_ids"].numel()
    print(tokens)


iteratorOfDl = iter(dataloader)
# 获取第一个batch
first_batch = next(iteratorOfDl)
print("\nFirst batch samples:")
for i in range(len(first_batch['act'])):
    print(f"Sample {i + 1}:")
    print(f"  act: {first_batch['act'][i]}")
    print(f"  prompt: {first_batch['prompt'][i]}")
    print("-" * 50)

# 获取第二个batch
second_batch = next(iteratorOfDl)
print("\nsecond batch samples:")

for i in range(len(second_batch['act'])):
    print(f"Sample {i + 11}:")
    print(f"  act: {second_batch['act'][i]}")
    print(f"  prompt: {second_batch['prompt'][i]}")
    print("-" * 50)
