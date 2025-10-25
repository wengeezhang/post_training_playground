from transformers import AutoTokenizer

# 下载模型文件到本地（如 ./deepseek_tokenizer）
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek_v3_tokenizer",  # 本地路径
    trust_remote_code=True
)

# 测试
text = "1+1="
# 编码文本 -> 得到 input_ids
input_ids = tokenizer.encode(text, return_tensors="pt")  # 返回PyTorch张量
print("Input IDs:", input_ids)
print("Decoded:", tokenizer.decode(input_ids[0]))  # 解码验证
