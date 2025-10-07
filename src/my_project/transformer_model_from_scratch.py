import torch
import torch.nn as nn

# 定义模型参数，与你之前的设定一致
d_model = 64
nhead = 4
num_decoder_layers = 3
dim_feedforward = 256
vocab_size = 10000  # 假设的词表大小
max_seq_length = 50  # 最大序列长度

# 1. 创建单个解码器层
decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    batch_first=True  # 设置输入张量形状为 (batch, seq, feature)
)

# 2. 堆叠多层形成解码器
transformer_decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=num_decoder_layers
)


# 3. 构建一个包含嵌入层和输出层的完整模型
class SimpleDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 使用一个可学习的位置编码，简单起见这里用Parameter
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.decoder = transformer_decoder
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        # tgt: 目标序列 (batch_size, tgt_seq_len)
        # memory: 编码器输出 (batch_size, src_seq_len, d_model)
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(d_model))  # 嵌入并缩放
        seq_len = tgt.size(1)
        tgt_emb = tgt_emb + self.pos_encoding[:, :seq_len, :]  # 添加位置编码

        # 生成因果掩码，防止解码器看到未来信息
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # 前向传播：关键步骤！
        # 将目标序列嵌入、编码器输出和因果掩码传入解码器
        decoder_output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # 将解码器输出映射到词表空间，得到每个位置的下一个词概率分布
        output_logits = self.output_layer(decoder_output)
        return output_logits


# 实例化模型
model = SimpleDecoderModel()

# 示例输入
batch_size = 2
tgt_seq_len = 10
src_seq_len = 15
tgt_dummy = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))  # 模拟目标输入（例如右移的标签）
memory_dummy = torch.randn(batch_size, src_seq_len, d_model)  # 模拟编码器输出

# 前向传播
output = model(tgt_dummy, memory_dummy)
print(f"输出形状: {output.shape}")  # 应为 (batch_size, tgt_seq_len, vocab_size)