import os
# 在导入任何模块之前设置虚拟的 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "dummy-key"
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

import torch
import torch.nn as nn
from transformers import TrainingArguments, PreTrainedTokenizer
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from transformers import PreTrainedModel, PretrainedConfig
import warnings
# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")


# 初始化组件
dpConfig = DeepseekV3Config(
    vocab_size=60,
    pad_token_id=56,
    eos_token_id=59,
    bos_token_id=58,

    hidden_size=32,
    num_attention_heads=8,
    num_key_value_heads=8,
    batch_first=True,  # 使用(batch, seq, feature)格式
    activation='relu'
)


model = DeepseekV3ForCausalLM(dpConfig)



print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
# 简单奖励函数

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# GRPO配置参数（调整为极简版本）

model.to(device)
model.eval()




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


dataset = create_simple_dataset()


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

tokenizer = SimpleTokenizer()

# get two example from dataset and tokenizer them ,then sent them to model

# Get 2 samples from the dataset
batch = dataset[:2]  # Get first 2 samples
print("Original samples:")
print(batch)

# Tokenize the batch
tokenized_batch = tokenizer(batch["prompt"])
print("\nTokenized batch:")
print(tokenized_batch)

model.num_key_value_groups = 1

# Pass through the model
with torch.no_grad():
    outputs = model(
        input_ids=tokenized_batch["input_ids"]
    )

print("\nModel outputs:")
print(f"Logits shape: {outputs.logits.shape}")  # Should be [batch_size, seq_len, vocab_size]

