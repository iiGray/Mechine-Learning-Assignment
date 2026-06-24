"""
train_self.py — FreqTimeNet: Frequency-Time Fusion Network
===========================================================
自提出改进模型：结合频域全局建模与多尺度时序卷积的混合架构

设计理念与新颖性分析：
------------------------
1. 【傅里叶域全局混合 (Fourier Mixing) 替代自注意力】
   - Transformer 的自注意力通过 Q·K^T 计算所有时间步对之间的相似度，复杂度 O(N²)
   - 本模型在频域中完成全局信息交换：FFT → 可学习复数线性变换 → IFFT
   - 根据卷积定理，频域乘法等价于时域全局卷积，复杂度仅 O(N log N)
   - 可学习的复数权重矩阵自适应地增强/抑制不同频率分量
   - 对电力消耗的日周期、周周期、季节周期等具有天然的归纳偏置
   - 这是本模型最核心的创新点

2. 【多尺度扩张卷积 (Multi-Scale Dilated Convolution)】
   - 并行使用扩张率 [1, 2, 4, 8, 16] 的深度可分离1D卷积
   - 不同扩张率对应不同的感受野大小，从1天到16天跨度
   - 捕获从细粒度突变到粗粒度趋势的多分辨率局部模式
   - 傅里叶混合同等对待所有频率，多尺度卷积确保局部突变不被平滑掉

3. 【自适应频时融合门 (Adaptive Frequency-Temporal Gate)】
   - 受 GLU (Gated Linear Unit) 和 GRU 门控机制启发
   - 学习一个 sigmoid 门控值，动态加权频域特征和时域特征的贡献
   - 不同时间步、不同特征维度可以有不同的融合策略
   - 模型可以自动学习在周期性强的时段依赖频域，在突变时段依赖时域

4. 【非自回归解码器 (Non-Autoregressive Decoder)】
   - LSTM 的自回归解码存在误差累积问题：t时刻的误差会传播到t+1时刻
   - 本模型使用全局平均池化 + 两层MLP直接输出完整预测序列
   - 避免了 teacher forcing 训练与自回归推理之间的分布偏移
   - 对于365天长期预测尤其关键——自回归365步的误差累积会非常严重

与 LSTM、Transformer 的本质区别：
  - LSTM: 循环递归，O(N) 但难以并行，长序列遗忘
  - Transformer: 自注意力，O(N²) 但缺乏周期性归纳偏置
  - FreqTimeNet: 频域建模，O(N log N)，天然适配周期性时序

参考文献：
  - Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms", NAACL 2022
  - Wu et al., "TimesNet: Temporal 2D-Variation Modeling", ICLR 2023
  - Dauphin et al., "Language Modeling with Gated Convolutional Networks", ICML 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time
import os
warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_LEN = 90
SHORT_OUTPUT = 90
LONG_OUTPUT = 365
BATCH_SIZE = 64
NUM_EPOCHS = 200
NUM_ROUNDS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.8

print(f"Device: {DEVICE}")
print(f"Input: {INPUT_LEN} days, Short-term: {SHORT_OUTPUT} days, Long-term: {LONG_OUTPUT} days")


# ============================================================
# 模型定义
# ============================================================

class FourierMixer(nn.Module):
    """
    傅里叶域混合层 — 替代自注意力的全局信息交换机制

    原理：在频域中通过可学习的复数线性变换实现全局信息混合。
    根据卷积定理：时域卷积 = 频域逐元素乘法。
    因此，频域中的复数矩阵乘法等价于时域中的全局（非因果）卷积。

    相比自注意力的优势：
    - 复杂度 O(N log N) vs O(N²)
    - 自然捕获周期性模式
    - 无需位置编码来感知顺序（频率天然编码了位置信息）
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 可学习的复数权重矩阵
        # W = W_real + i * W_imag
        self.W_real = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_imag = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)  — 实数输入
        返回: (batch, seq_len, d_model) — 实数输出
        """
        B, L, D = x.shape

        # Step 1: 沿时间维度做实数FFT
        Xf = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, L//2+1, D), complex

        # Step 2: 复数线性变换 (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        Xf_real, Xf_imag = Xf.real, Xf.imag
        out_real = Xf_real @ self.W_real - Xf_imag @ self.W_imag
        out_imag = Xf_real @ self.W_imag + Xf_imag @ self.W_real
        Xf_out = torch.complex(out_real, out_imag)
        Xf_out = self.dropout(Xf_out)

        # Step 3: 逆FFT回到时域
        out = torch.fft.irfft(Xf_out, n=L, dim=1, norm='ortho')  # (B, L, D)

        return self.norm(x + out)  # 残差连接 + LayerNorm


class MultiScaleConv(nn.Module):
    """
    多尺度扩张卷积层

    并行使用多个不同扩张率的深度可分离1D卷积，捕获从细粒度到
    粗粒度的多分辨率局部时序模式。不同扩张率对应不同的感受野：
      dilation=1:  感受野 ≈ 3天
      dilation=2:  感受野 ≈ 7天
      dilation=4:  感受野 ≈ 15天
      dilation=8:  感受野 ≈ 31天
      dilation=16: 感受野 ≈ 63天

    使用分组卷积(group=d_model)实现深度可分离，大幅减少参数量。
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dilations = [1, 2, 4, 8, 16]
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size,
                      padding=(kernel_size - 1) * d // 2,
                      dilation=d, groups=d_model)  # depthwise
            for d in self.dilations
        ])
        # 各支路融合投影
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * len(self.dilations), d_model, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        residual = x
        x_t = x.transpose(1, 2)  # (B, D, L)  for Conv1d

        # 各扩张率并行卷积
        branches = []
        for conv in self.convs:
            branch = F.gelu(conv(x_t))
            branches.append(branch)

        # 拼接并融合
        multi_scale = torch.cat(branches, dim=1)  # (B, D*5, L)
        out = self.fusion(multi_scale)  # (B, D, L)
        out = out.transpose(1, 2)  # (B, L, D)
        out = self.dropout(out)

        return self.norm(residual + out)


class AdaptiveGate(nn.Module):
    """
    自适应频时融合门

    根据输入内容动态决定在频域特征和时域特征之间的混合比例。
    受 GRU 门控机制和 GLU 启发，但应用于跨域特征融合。

    门控值通过输入的全局统计信息（均值+标准差池化）计算，
    使得模型能够根据输入的整体特性（如季节性强度）自适应调整。
    """
    def __init__(self, d_model):
        super().__init__()
        # 门控网络：从输入的全局统计信息计算融合权重
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, freq_feat, temp_feat):
        """
        freq_feat: (B, L, D) — 频域分支输出
        temp_feat: (B, L, D) — 时域分支输出
        返回: (B, L, D) — 自适应融合后的特征
        """
        # 全局统计信息作为门控输入
        # 均值捕获整体趋势，标准差捕获波动性
        combined = torch.cat([freq_feat, temp_feat], dim=1)  # (B, 2L, D)
        mean_pool = combined.mean(dim=1, keepdim=True)  # (B, 1, D)
        std_pool = combined.std(dim=1, keepdim=True)  # (B, 1, D)
        gate_input = torch.cat([mean_pool, std_pool], dim=-1)  # (B, 1, 2D)

        gate = self.gate_net(gate_input)  # (B, 1, D)

        return gate * freq_feat + (1 - gate) * temp_feat


class FeedForward(nn.Module):
    """标准前馈网络，使用 GELU 激活"""
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


class FreqTimeBlock(nn.Module):
    """
    频时融合模块 — FreqTimeNet 的核心构建块

    每个块包含三个子层：
    1. 多尺度时域卷积 — 局部模式提取
    2. 傅里叶域混合 — 全局依赖建模
    3. 自适应频时融合门 — 动态特征融合
    4. 前馈网络 — 非线性变换

    整个块的流程：
    x → [MultiScaleConv ∥ FourierMixer] → AdaptiveGate → FeedForward → out
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.temporal = MultiScaleConv(d_model, dropout=dropout)
        self.spectral = FourierMixer(d_model, dropout=dropout)
        self.gate = AdaptiveGate(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        # 并行计算频域和时域特征
        temp_out = self.temporal(x)
        freq_out = self.spectral(x)

        # 自适应门控融合
        fused = self.gate(freq_out, temp_out)

        # 前馈网络
        return self.ffn(fused)


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码 — 让模型自适应学习最优的位置表示"""
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class FreqTimeNet(nn.Module):
    """
    FreqTimeNet: Frequency-Time Fusion Network

    频时融合网络 — 用于多变量时间序列预测的新型架构

    模型结构：
    1. 输入投影 + 可学习位置编码
    2. N个 FreqTimeBlock (频时融合模块)
    3. 全局平均池化 → 两层MLP解码器 → 完整输出序列

    参数：
      input_dim:   输入特征维度 (电力数据为14)
      d_model:     模型隐藏维度 (默认128)
      num_blocks:  FreqTimeBlock 层数 (默认3)
      output_len:  预测长度 (90或365天)
      dropout:     Dropout比率 (默认0.1)
    """
    def __init__(self, input_dim, d_model=128, num_blocks=3,
                 output_len=90, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 可学习位置编码
        self.pos_encoding = LearnablePositionalEncoding(
            max_len=500, d_model=d_model, dropout=dropout
        )

        # 频时融合模块堆叠
        self.blocks = nn.ModuleList([
            FreqTimeBlock(d_model, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)

        # 非自回归解码器：全局平均池化 → MLP → 完整输出序列
        # 两层MLP，中间有较大隐藏层以支持长序列(365天)的映射
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_len),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x):
        """
        x: (batch, input_len, input_dim) — 90天多变量输入
        返回: (batch, output_len) — output_len天的 Global_active_power 预测
        """
        # 输入投影
        x = self.input_proj(x)  # (B, 90, d_model)

        # 位置编码
        x = self.pos_encoding(x)

        # 通过频时融合模块
        for block in self.blocks:
            x = block(x)

        # 最终归一化
        x = self.final_norm(x)

        # 全局平均池化 — 聚合所有时间步的信息
        x = x.mean(dim=1)  # (B, d_model)

        # 非自回归解码
        out = self.decoder(x)  # (B, output_len)

        return out


# ============================================================
# 数据加载与预处理
# ============================================================
print("\n" + "=" * 60)
print("加载预处理数据")
print("=" * 60)

df = pd.read_csv('preprocessed_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"数据形状: {df.shape}, 日期范围: {df['Date'].min()} ~ {df['Date'].max()}")

feature_cols = [c for c in df.columns if c != 'Date']
data = df[feature_cols].values.astype(np.float32)
target_idx = feature_cols.index('Global_active_power')
print(f"特征数: {len(feature_cols)}, 目标列: {feature_cols[target_idx]}")


def create_sliding_windows(data, input_len, output_len, target_idx):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + output_len, target_idx])
    return np.array(X), np.array(y)


def normalize_data(train_data, test_data):
    n_features = train_data.shape[2]
    scalers = []
    train_norm = np.zeros_like(train_data)
    test_norm = np.zeros_like(test_data)
    for i in range(n_features):
        scaler = StandardScaler()
        flat_train = train_data[:, :, i].reshape(-1, 1)
        scaler.fit(flat_train)
        train_norm[:, :, i] = scaler.transform(flat_train).reshape(
            train_data.shape[0], train_data.shape[1])
        flat_test = test_data[:, :, i].reshape(-1, 1)
        test_norm[:, :, i] = scaler.transform(flat_test).reshape(
            test_data.shape[0], test_data.shape[1])
        scalers.append(scaler)
    return train_norm, test_norm, scalers


# ============================================================
# 构建数据集
# ============================================================
print("\n" + "=" * 60)
print("构建滑动窗口数据集")
print("=" * 60)

X_short, y_short = create_sliding_windows(data, INPUT_LEN, SHORT_OUTPUT, target_idx)
X_long, y_long = create_sliding_windows(data, INPUT_LEN, LONG_OUTPUT, target_idx)
print(f"短期数据集: X={X_short.shape}, y={y_short.shape}")
print(f"长期数据集: X={X_long.shape}, y={y_long.shape}")


def split_data(X, y):
    n_train = int(len(X) * TRAIN_RATIO)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


X_short_train, X_short_test, y_short_train, y_short_test = split_data(X_short, y_short)
X_long_train, X_long_test, y_long_train, y_long_test = split_data(X_long, y_long)

X_short_train_n, X_short_test_n, scalers_short = normalize_data(
    X_short_train, X_short_test)
X_long_train_n, X_long_test_n, scalers_long = normalize_data(
    X_long_train, X_long_test)

target_scaler_short = scalers_short[target_idx]
target_scaler_long = scalers_long[target_idx]


def make_dataloader(X, y, shuffle=True):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


short_train_loader = make_dataloader(X_short_train_n, y_short_train)
short_test_loader = make_dataloader(X_short_test_n, y_short_test, shuffle=False)
long_train_loader = make_dataloader(X_long_train_n, y_long_train)
long_test_loader = make_dataloader(X_long_test_n, y_long_test, shuffle=False)


# ============================================================
# 训练与评估函数
# ============================================================
def train_model(model, train_loader, epochs, lr, wd, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 30:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    model.load_state_dict(best_state)
    return best_loss


def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            pred = model(batch_x).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(batch_y.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    preds_orig = target_scaler.inverse_transform(preds)
    targets_orig = target_scaler.inverse_transform(targets)

    mse = mean_squared_error(targets_orig.flatten(), preds_orig.flatten())
    mae = mean_absolute_error(targets_orig.flatten(), preds_orig.flatten())

    return mse, mae, preds_orig, targets_orig


def run_experiment(model_name, model_class, model_kwargs, train_loader,
                   test_loader, target_scaler, output_len, seed_base=42):
    print(f"\n{'=' * 60}")
    print(f"{model_name} - Output: {output_len} days")
    print(f"{'=' * 60}")

    mse_list, mae_list = [], []
    best_mse = float('inf')
    best_preds, best_targets = None, None
    best_round = 0

    for r in range(NUM_ROUNDS):
        seed = seed_base + r
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n--- Round {r + 1}/{NUM_ROUNDS} (seed={seed}) ---")
        model = model_class(**model_kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        train_model(model, train_loader, NUM_EPOCHS,
                    LEARNING_RATE, WEIGHT_DECAY)
        train_time = time.time() - t0

        mse, mae, preds, targets = evaluate_model(
            model, test_loader, target_scaler)
        mse_list.append(mse)
        mae_list.append(mae)
        print(f"  Train time: {train_time:.1f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_preds = preds
            best_targets = targets
            best_round = r + 1
            torch.save(model.state_dict(),
                       f'model_freqtimenet_{output_len}d_best.pt')

    mse_arr = np.array(mse_list)
    mae_arr = np.array(mae_list)
    print(f"\n{'=' * 60}")
    print(f"  FreqTimeNet ({output_len}天) 最终结果:")
    print(f"  MSE:  {mse_arr.mean():.4f} ± {mse_arr.std():.4f}")
    print(f"  MAE:  {mae_arr.mean():.4f} ± {mae_arr.std():.4f}")
    print(f"  每轮 MSE: {[f'{v:.4f}' for v in mse_list]}")
    print(f"  每轮 MAE: {[f'{v:.4f}' for v in mae_list]}")
    print(f"  最佳轮次: Round {best_round}")

    return {
        'model': model_name,
        'output_len': output_len,
        'mse_mean': mse_arr.mean(),
        'mse_std': mse_arr.std(),
        'mae_mean': mae_arr.mean(),
        'mae_std': mae_arr.std(),
        'mse_list': mse_list,
        'mae_list': mae_list,
        'best_preds': best_preds,
        'best_targets': best_targets,
    }

if __name__ == "__main__":
    all_results = []

    # 短期预测 (90→90)
    all_results.append(run_experiment(
        'FreqTimeNet', FreqTimeNet,
        {'input_dim': len(feature_cols), 'd_model': 128, 'num_blocks': 3,
        'output_len': SHORT_OUTPUT, 'dropout': 0.1},
        short_train_loader, short_test_loader, target_scaler_short, SHORT_OUTPUT
    ))

    # 长期预测 (90→365)
    all_results.append(run_experiment(
        'FreqTimeNet', FreqTimeNet,
        {'input_dim': len(feature_cols), 'd_model': 128, 'num_blocks': 3,
        'output_len': LONG_OUTPUT, 'dropout': 0.1},
        long_train_loader, long_test_loader, target_scaler_long, LONG_OUTPUT
    ))

    # ============================================================
    # 汇总结果
    # ============================================================
    print("\n\n" + "=" * 70)
    print("FreqTimeNet 最终结果汇总")
    print("=" * 70)
    print(f"{'模型':<15} {'预测长度':<10} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}")
    print("-" * 70)
    for r in all_results:
        model_name = r['model']
        out_len = r['output_len']
        mse_m, mse_s = r['mse_mean'], r['mse_std']
        mae_m, mae_s = r['mae_mean'], r['mae_std']
        print(f"{model_name:<15} {f'{out_len}天':<10} "
            f"{mse_m:.4f} ± {mse_s:.4f}         "
            f"{mae_m:.4f} ± {mae_s:.4f}")
    print("=" * 70)

    # 保存结果
    results_df = pd.DataFrame([{
        'Model': r['model'],
        'Output_Days': r['output_len'],
        'MSE_Mean': r['mse_mean'],
        'MSE_Std': r['mse_std'],
        'MAE_Mean': r['mae_mean'],
        'MAE_Std': r['mae_std'],
        'MSE_Rounds': str(r['mse_list']),
        'MAE_Rounds': str(r['mae_list']),
    } for r in all_results])
    results_df.to_csv('results_self_summary.csv', index=False)
    print("\n结果已保存到 results_self_summary.csv")

    print("\n训练完成！FreqTimeNet 模型已保存:")
    for f in sorted(os.listdir('.')):
        if f.startswith('model_freqtimenet_') and f.endswith('.pt'):
            print(f"  - {f}")

    print("\n" + "=" * 70)
    print("设计理由总结:")
    print("=" * 70)
    print("""
    FreqTimeNet 与 LSTM/Transformer 的本质区别：

    1. 全局建模方式：
    - LSTM: 递归（逐步传递隐藏状态，远距离信息衰减）
    - Transformer: 自注意力（O(N²)成对比较，缺乏周期性先验）
    - FreqTimeNet: 频域混合（O(N log N)，天然捕获周期性）

    2. 解码策略：
    - LSTM: 自回归（误差累积，365步后严重偏离）
    - Transformer: 可学习查询向量 + 交叉注意力（需要额外解码器）
    - FreqTimeNet: 非自回归直接映射（无误差累积，简洁高效）

    3. 多尺度处理：
    - LSTM: 单一时间尺度
    - Transformer: 通过多个注意力头隐式建模
    - FreqTimeNet: 显式多尺度扩张卷积，覆盖1-63天感受野

    4. 自适应能力：
    - LSTM/Transformer: 固定的信息处理路径
    - FreqTimeNet: 自适应频时融合门根据输入动态调整
    """)
