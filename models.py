"""
LSTM 和 Transformer 模型定义
用于多变量时间序列预测：输入90天 → 输出90天(短期)或365天(长期)
"""
import torch
import torch.nn as nn
import math


class LSTMPredictor(nn.Module):
    """LSTM 编码器-解码器预测模型"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_len=90, dropout=0.2):
        super().__init__()
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            1, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        _, (hidden, cell) = self.encoder(x)

        decoder_input = x[:, -1, 0:1].unsqueeze(1)  # (batch, 1, 1)
        outputs = []

        for t in range(self.output_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.dropout(out)
            pred = self.fc(out)  # (batch, 1, 1)
            outputs.append(pred.squeeze(-1))
            decoder_input = pred  # 自回归

        return torch.stack(outputs, dim=1).squeeze(-1)  # (batch, output_len)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3,
                 output_len=90, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出查询向量（可学习的）
        self.output_query = nn.Parameter(torch.randn(1, output_len, d_model) * 0.1)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, input_len, input_dim)
        返回: (batch, output_len)
        """
        batch_size = x.size(0)

        # 编码输入
        x = self.input_proj(x)  # (batch, input_len, d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)

        # 解码: 使用可学习的查询向量
        queries = self.output_query.expand(batch_size, -1, -1)
        queries = self.pos_encoder(queries)
        out = self.transformer_decoder(queries, memory)
        out = self.output_proj(out).squeeze(-1)  # (batch, output_len)

        return out
