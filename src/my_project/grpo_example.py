import os
# 在导入任何模块之前设置虚拟的 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "dummy-key"

import torch
from transformers import TrainingArguments, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


# 超小型模型配置
class TinyModelConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size=1000, hidden_size=64, num_hidden_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.eos_token_id = 103  # 与Tokenizer中的EOS ID一致


# 极简Transformer模型（仅用于演示）
class TinyModel(PreTrainedModel):
    config_class = TinyModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, logits_to_keep=None, **kwargs):
        # 处理 logits_to_keep 参数
        if logits_to_keep is not None:
            # GRPOTrainer 会先 +1 再 -1，所以我们需要正确处理这个逻辑
            # 如果指定了 logits_to_keep，只保留前 logits_to_keep 个 token
            if logits_to_keep < input_ids.shape[1]:
                input_ids = input_ids[:, :logits_to_keep]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :logits_to_keep]
                if labels is not None and labels.shape[1] > logits_to_keep:
                    labels = labels[:, :logits_to_keep]

        # 确保输入输出长度一致
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.tanh(layer(x))
        logits = self.lm_head(x)  # shape: (batch_size, seq_len, vocab_size)

        # 确保输出长度与输入一致
        if attention_mask is not None:
            # 将填充位置的logits设为极小值
            logits = logits.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))

        loss = None
        if labels is not None:
            # 确保labels和logits的序列长度一致
            if labels.shape[1] != logits.shape[1]:
                min_len = min(labels.shape[1], logits.shape[1])
                labels = labels[:, :min_len]
                logits = logits[:, :min_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :min_len]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 返回符合 GRPOTrainer 期望的格式
        return type('', (), {'loss': loss, 'logits': logits})()

    def generate(self, input_ids, max_length=20, **kwargs):
        batch_size = input_ids.shape[0]
        current_ids = input_ids
        all_eos = torch.zeros_like(input_ids, dtype=torch.bool)  # 初始化EOS标记

        for _ in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.forward(current_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), num_samples=1
                )

                # 标记哪些序列已经生成EOS
                eos_mask = (next_tokens == self.config.eos_token_id)
                all_eos = torch.cat([all_eos, eos_mask], dim=1)

                # 对于已经结束的序列，继续填充EOS
                next_tokens[all_eos.any(dim=1)] = self.config.eos_token_id

            current_ids = torch.cat([current_ids, next_tokens], dim=1)

        return current_ids


# 创建10个简单样本的数据集
def create_simple_dataset():
    samples = [
        {"prompt": "1+1=", "response": "2"},  # 将 query 改为 prompt
        {"prompt": "2+2=", "response": "4"},
        {"prompt": "3+3=", "response": "6"},
        {"prompt": "4+4=", "response": "8"},
        {"prompt": "5+5=", "response": "10"},
        {"prompt": "6+6=", "response": "12"},
        {"prompt": "7+7=", "response": "14"},
        {"prompt": "8+8=", "response": "16"},
        {"prompt": "9+9=", "response": "18"},
        {"prompt": "10+10=", "response": "20"},
        {"prompt": "11+11=", "response": "22"},  # 将 query 改为 prompt
        {"prompt": "12+12=", "response": "24"},
        {"prompt": "13+13=", "response": "26"},
        {"prompt": "14+14=", "response": "28"},
        {"prompt": "15+15=", "response": "30"},
        {"prompt": "16+16=", "response": "32"},
        {"prompt": "17+17=", "response": "34"},
        {"prompt": "18+18=", "response": "36"},
        {"prompt": "19+19=", "response": "38"},
        {"prompt": "20+20=", "response": "40"}
    ]
    return Dataset.from_list(samples)

train_dataset = create_simple_dataset()
print(f"数据集大小: {len(train_dataset)}")
print(f"样本示例: {train_dataset[0]}")


class SimpleTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        # 创建简单的词汇表
        vocab = {str(i): i for i in range(100)}
        vocab.update({"[PAD]": 100, "[UNK]": 101, "[BOS]": 102, "[EOS]": 103})
        # 设置词汇表相关属性
        self.vocab = vocab.copy()

        # 先初始化父类
        super().__init__(**kwargs)
        self._vocab_str_to_int = self.vocab
        self._vocab_int_to_str = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 103  # 明确设置EOS token ID
        self.pad_token_id = 100

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, text):
        # 简单分词：按字符分割
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["[UNK]"])

    def _convert_id_to_token(self, index):
        return self._vocab_int_to_str.get(index, "[UNK]")

    def encode(self, text, **kwargs):
        tokens = self._tokenize(text)
        ids = [self.vocab["[BOS]"]] + [self._convert_token_to_id(t) for t in tokens] + [self.vocab["[EOS]"]]
        return ids

    def decode(self, token_ids, **kwargs):
        tokens = [self._convert_id_to_token(id) for id in token_ids if
                  id not in [self.vocab["[BOS]"], self.vocab["[EOS]"]]]
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """保存词汇表到文件"""
        import os
        import json

        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)

        # 构建文件名
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        # 保存词汇表
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            encoded = self.encode(text)
            return {
                "input_ids": torch.tensor([encoded]),
                "attention_mask": torch.tensor([[1] * len(encoded)])  # 添加attention_mask
            }
        elif isinstance(text, dict) and "prompt" in text:
            encoded = self.encode(text["prompt"])
            return {
                "input_ids": torch.tensor([encoded]),
                "attention_mask": torch.tensor([[1] * len(encoded)])
            }
        else:
            # 处理批量文本
            encoded_list = [self.encode(t) for t in text]
            max_len = max(len(seq) for seq in encoded_list)

            # 填充并创建attention_mask
            padded_ids = []
            attention_masks = []
            for seq in encoded_list:
                pad_len = max_len - len(seq)
                padded_seq = seq + [self.pad_token_id] * pad_len
                padded_ids.append(padded_seq)
                attention_masks.append([1] * len(seq) + [0] * pad_len)

            return {
                "input_ids": torch.tensor(padded_ids),
                "attention_mask": torch.tensor(attention_masks)
            }

# 初始化组件
tokenizer = SimpleTokenizer()
model_config = TinyModelConfig(vocab_size=104, hidden_size=32, num_hidden_layers=1)
model = TinyModel(model_config)

print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# GRPO配置参数（调整为极简版本）
args = GRPOConfig(
    per_device_train_batch_size=2,
    #generation_batch_size=4,
    num_generations=2,
    num_iterations=2,
    steps_per_generation=4,
    max_completion_length=10,
    learning_rate=1e-4,
    gradient_accumulation_steps=2,
    max_steps=5,
    output_dir="./tiny_grpo_output",
    seed=42,
    logging_steps=1,
    save_steps=10,
    fp16=False,  # 禁用 fp16
    bf16=False,  # 禁用 bf16
    use_vllm=False,
    logging_dir=None,  # 禁用日志目录
    report_to=[],     # 空列表表示不向任何集成报告
)

# 简单奖励函数
def custom_reward_func(prompts=None, completions=None, completion_ids=None, **kwargs):
    rewards = []

    # 使用 completions 参数，这是模型生成的内容
    if completions is not None:
        samples = completions
    else:
        # 备用方案
        samples = prompts if prompts is not None else []

    for sample in samples:
        # 处理样本文本
        if isinstance(sample, dict):
            text = sample.get("response", sample.get("completion", ""))
        else:
            text = str(sample)

        # 简单奖励逻辑：如果包含数字则给高分
        if any(c.isdigit() for c in text):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# 初始化GRPOTrainer
# 正确的 GRPOTrainer 初始化
trainer = GRPOTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    reward_funcs=custom_reward_func,
    processing_class=tokenizer,  # 关键修正：使用 processing_class
    callbacks=[],  # 禁用所有回调
)

print("开始训练...")
trainer.train()
print("训练完成!")