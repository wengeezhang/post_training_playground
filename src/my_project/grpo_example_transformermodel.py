import os
# 在导入任何模块之前设置虚拟的 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "dummy-key"

import torch
import torch.nn as nn
from transformers import TrainingArguments, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from transformers import PreTrainedModel, PretrainedConfig


# 超小型模型配置
class TinyModelConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 dim_feedforward=256, max_position_embeddings=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_feedforward = dim_feedforward
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = 59
        self.pad_token_id = 56


# 极简Transformer模型（仅用于演示）
# 使用TransformerDecoder改造的模型
class TinyModel(PreTrainedModel):
    config_class = TinyModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # 位置编码（可学习的）
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.hidden_size)
        )

        # 创建Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,  # 使用(batch, seq, feature)格式
            activation='relu'
        )

        # 创建Transformer解码器（堆叠多层）
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_hidden_layers
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def _generate_causal_mask(self, seq_len, device):
        """生成因果掩码，防止解码器看到未来信息[3,5](@ref)"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)

    def forward(self, input_ids, attention_mask=None, labels=None, logits_to_keep=None, **kwargs):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 词嵌入 + 位置编码
        x = self.embedding(input_ids)  # (batch_size, seq_len, hidden_size)
        x = x + self.positional_encoding[:, :seq_len, :]

        # 生成因果掩码[3](@ref)
        causal_mask = self._generate_causal_mask(seq_len, device)

        # 由于是纯解码器架构，memory和tgt使用相同的输入[2,5](@ref)
        # 这里memory=tgt，实现自回归解码
        decoder_output = self.decoder(
            tgt=x,  # 目标序列
            memory=x,  # 编码器输出（这里用相同的输入模拟纯解码器架构）
            tgt_mask=causal_mask,  # 因果掩码
            tgt_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        lm_head_inputs = decoder_output
        if logits_to_keep is not None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            lm_head_inputs = decoder_output[:, slice_indices, :]
        logits = self.lm_head(lm_head_inputs)  # (batch_size, seq_len, vocab_size)

        # 处理注意力掩码（保持原有逻辑）
        # 处理 logits_to_keep 参数（保持原有逻辑）
        if logits_to_keep is not None:
            if logits_to_keep < input_ids.shape[1]:
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :logits_to_keep]
                if labels is not None and labels.shape[1] > logits_to_keep:
                    labels = labels[:, :logits_to_keep]

        if attention_mask is not None:
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
        """生成文本（保持原有逻辑，但使用新的Transformer结构）[5](@ref)"""
        batch_size = input_ids.shape[0]
        current_ids = input_ids
        all_eos = torch.zeros_like(input_ids, dtype=torch.bool)

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
        {"prompt": "20+20=", "response": "40"},
        {"prompt": "21+21=", "response": "42"},  # 将 query 改为 prompt
        {"prompt": "22+22=", "response": "44"},
        {"prompt": "23+23=", "response": "46"},
        {"prompt": "24+24=", "response": "48"},
        {"prompt": "25+25=", "response": "50"},
        {"prompt": "26+26=", "response": "52"},
        {"prompt": "27+27=", "response": "54"},
        {"prompt": "28+28=", "response": "56"},
        {"prompt": "29+29=", "response": "58"},
        {"prompt": "30+30=", "response": "60"},
        {"prompt": "31+31=", "response": "62"},  # 将 query 改为 prompt
        {"prompt": "32+32=", "response": "64"},
        {"prompt": "33+33=", "response": "66"},
        {"prompt": "34+34=", "response": "68"},
        {"prompt": "35+35=", "response": "70"},
        {"prompt": "36+36=", "response": "72"},
        {"prompt": "37+37=", "response": "74"},
        {"prompt": "38+38=", "response": "76"},
        {"prompt": "39+39=", "response": "78"},
    ]
    return Dataset.from_list(samples)


train_dataset = create_simple_dataset()
print(f"数据集大小: {len(train_dataset)}")
print(f"样本示例: {train_dataset[0]}")


class SimpleTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        # 创建简单的词汇表
        # 修改词汇表初始化部分
        vocab = {
            # 数字 0-9
            **{str(i): i for i in range(10)},

            # 数学符号 (10-19)
            "+": 10,
            "=": 11,
            "-": 12,
            "*": 13,
            "/": 14,
            "(": 15,
            ")": 16,
            "[": 17,
            "]": 18,
            "{": 19,

            # 字母 a-z (20-45)
            **{chr(97 + i): 20 + i for i in range(26)},

            # 空格和标点 (46-55)
            " ": 46,
            ".": 47,
            ",": 48,
            ":": 49,
            ";": 50,
            "?": 51,
            "!": 52,
            "'": 53,
            '"': 54,
            "\\": 55,

            # 特殊token (56-59)
            "[PAD]": 56,
            "[UNK]": 57,
            "[BOS]": 58,
            "[EOS]": 59
        }
        # 设置词汇表相关属性
        self.vocab = vocab.copy()

        # 先初始化父类
        super().__init__(**kwargs)
        self._vocab_str_to_int = self.vocab
        self._vocab_int_to_str = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 59
        self.pad_token_id = 56
        self.unk_token_id = 57
        self.bos_token_id = 58

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
                "input_ids": torch.tensor([encoded]).to(device),
                "attention_mask": torch.tensor([[1] * len(encoded)]).to(device)
            }
        elif isinstance(text, dict) and "prompt" in text:
            encoded = self.encode(text["prompt"])
            return {
                "input_ids": torch.tensor([encoded]).to(device),
                "attention_mask": torch.tensor([[1] * len(encoded)]).to(device)
            }
        else:
            encoded_list = [self.encode(t) for t in text]
            max_len = max(len(seq) for seq in encoded_list)

            padded_ids = []
            attention_masks = []
            for seq in encoded_list:
                pad_len = max_len - len(seq)
                padded_seq = seq + [self.pad_token_id] * pad_len
                padded_ids.append(padded_seq)
                attention_masks.append([1] * len(seq) + [0] * pad_len)

            return {
                "input_ids": torch.tensor(padded_ids).to(device),
                "attention_mask": torch.tensor(attention_masks).to(device)
            }


# 初始化组件
tokenizer = SimpleTokenizer()
model_config = TinyModelConfig(vocab_size=60, hidden_size=32, num_hidden_layers=1)
model = TinyModel(model_config)

print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
# 简单奖励函数

# forward before training


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


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# GRPO配置参数（调整为极简版本）
args = GRPOConfig(
    per_device_train_batch_size=3,
    #generation_batch_size=4,
    num_generations=2,
    num_iterations=2,
    steps_per_generation=6,
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

model.to(device)

# 测试tokenizer
test_str = "1+1="
print(f"原始字符串: {test_str}")
encoded = tokenizer.encode(test_str)
print(f"编码结果: {encoded}")
decoded = tokenizer.decode(encoded)
print(f"解码结果: {decoded}") # 输出应为 "1+1="


sample_prompt_after = "1+1="
sample_inputs_after = tokenizer(sample_prompt_after, return_tensors="pt")
sample_inputs_after = {k: v.to(device) for k, v in sample_inputs_after.items()}

with torch.no_grad():
    sample_outputs_after = model(**sample_inputs_after)
    print(f"模型输出 shape: {sample_outputs_after.logits.shape}")

    pred_token_ids_after = torch.argmax(sample_outputs_after.logits, dim=-1)
    print(f"预测的token ID: {pred_token_ids_after}")
    pred_tokens_after = tokenizer.decode(pred_token_ids_after[0].cpu().numpy())
    print(f"预测的token: {pred_tokens_after}")