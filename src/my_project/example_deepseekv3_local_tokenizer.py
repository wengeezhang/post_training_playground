import os
# 在导入任何模块之前设置虚拟的 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "dummy-key"
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

import torch
import torch.nn as nn
from transformers import TrainingArguments, PreTrainedTokenizer, AutoTokenizer
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
    vocab_size=129280,
    bos_token_id=0,  # 起始token ID
    eos_token_id=1,  # 结束token ID
    pad_token_id=2,

    hidden_size=4096,  # 建议增大到合理值
    num_attention_heads=8,
    num_key_value_heads=8,
    batch_first=True,  # 使用(batch, seq, feature)格式
    activation='relu'
)


model = DeepseekV3ForCausalLM(dpConfig)

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


tokenizer = AutoTokenizer.from_pretrained(
    "deepseek_v3_tokenizer",  # 本地路径
    trust_remote_code=True
)

# get two example from dataset and tokenizer them ,then sent them to model

input_ids = tokenizer.encode("1+1=", return_tensors="pt")

# Tokenize the batch

# Pass through the model
with torch.no_grad():
    outputs = model(
        input_ids=input_ids
    )

print("\nModel outputs:")
print(f"Logits shape: {outputs.logits.shape}")  # Should be [batch_size, seq_len, vocab_size]

# decode outputs.logits

pred_token_ids_after = torch.argmax(outputs.logits, dim=-1)
print(f"预测的token ID: {pred_token_ids_after}")
pred_tokens_after_0 = tokenizer.decode(pred_token_ids_after[0].cpu().numpy())
pred_tokens_after_1 = tokenizer.decode(pred_token_ids_after[1].cpu().numpy())
print(f"预测的token: {pred_tokens_after_0}")
print(f"预测的token: {pred_tokens_after_1}")

