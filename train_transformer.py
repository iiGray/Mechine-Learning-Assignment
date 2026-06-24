"""
train_transformer.py — Transformer 时间序列预测训练脚本
=========================================================
短期预测：90天输入 → 90天输出
长期预测：90天输入 → 365天输出
各5轮实验，报告 MSE/MAE 均值和标准差
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

from models import TransformerPredictor

# ============================================================
# 配置
# ============================================================
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

print("\n" + "="*60)
print("加载预处理数据")
print("="*60)

df = pd.read_csv('preprocessed_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"数据形状: {df.shape}, 日期范围: {df['Date'].min()} ~ {df['Date'].max()}")

feature_cols = [c for c in df.columns if c != 'Date']
data = df[feature_cols].values.astype(np.float32)

target_idx = feature_cols.index('Global_active_power')
print(f"特征数: {len(feature_cols)}, 目标列: {feature_cols[target_idx]}")
print(f"特征列: {feature_cols}")


def create_sliding_windows(data, input_len, output_len, target_idx):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, target_idx])
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


def train_model(model, train_loader, epochs, lr, wd, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    loss_history = []

    for epoch in tqdm(range(epochs)):
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
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 30:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    model.load_state_dict(best_state)
    return best_loss, loss_history


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


# ============================================================
# 构建数据集
# ============================================================
print("\n" + "="*60)
print("构建滑动窗口数据集")
print("="*60)

X_short, y_short = create_sliding_windows(data, INPUT_LEN, SHORT_OUTPUT, target_idx)
X_long, y_long = create_sliding_windows(data, INPUT_LEN, LONG_OUTPUT, target_idx)
print(f"短期数据集: X={X_short.shape}, y={y_short.shape}")
print(f"长期数据集: X={X_long.shape}, y={y_long.shape}")


def split_data(X, y):
    n_train = int(len(X) * TRAIN_RATIO)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


X_short_train, X_short_test, y_short_train, y_short_test = split_data(X_short, y_short)
X_long_train, X_long_test, y_long_train, y_long_test = split_data(X_long, y_long)

print(f"短期 - 训练: {X_short_train.shape[0]}, 测试: {X_short_test.shape[0]}")
print(f"长期 - 训练: {X_long_train.shape[0]}, 测试: {X_long_test.shape[0]}")

# 标准化
X_short_train_n, X_short_test_n, scalers_short = normalize_data(X_short_train, X_short_test)
X_long_train_n, X_long_test_n, scalers_long = normalize_data(X_long_train, X_long_test)

target_scaler_short = scalers_short[target_idx]
target_scaler_long = scalers_long[target_idx]

# 归一化目标值 y
y_short_train_n = target_scaler_short.transform(
    y_short_train.reshape(-1, 1)).reshape(y_short_train.shape)
y_short_test_n = target_scaler_short.transform(
    y_short_test.reshape(-1, 1)).reshape(y_short_test.shape)
y_long_train_n = target_scaler_long.transform(
    y_long_train.reshape(-1, 1)).reshape(y_long_train.shape)
y_long_test_n = target_scaler_long.transform(
    y_long_test.reshape(-1, 1)).reshape(y_long_test.shape)

print(f"y_short_train raw range: [{y_short_train.min():.2f}, {y_short_train.max():.2f}]")
print(f"y_short_train norm range: [{y_short_train_n.min():.2f}, {y_short_train_n.max():.2f}]")
print(f"y_long_train raw range: [{y_long_train.min():.2f}, {y_long_train.max():.2f}]")
print(f"y_long_train norm range: [{y_long_train_n.min():.2f}, {y_long_train_n.max():.2f}]")


def make_dataloader(X, y, shuffle=True):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


short_train_loader = make_dataloader(X_short_train_n, y_short_train_n)
short_test_loader = make_dataloader(X_short_test_n, y_short_test_n, shuffle=False)
long_train_loader = make_dataloader(X_long_train_n, y_long_train_n)
long_test_loader = make_dataloader(X_long_test_n, y_long_test_n, shuffle=False)


# ============================================================
# 实验运行
# ============================================================
def run_experiment(model_name, model_class, model_kwargs, train_loader, test_loader,
                   target_scaler, output_len, seed_base=42):
    print(f"\n{'='*60}")
    print(f"{model_name} - Output: {output_len} days")
    print(f"{'='*60}")

    mse_list, mae_list = [], []
    best_mse = float('inf')
    best_mae = float('inf')
    best_preds, best_targets = None, None
    best_loss_history = None
    best_round = 0
    model_slug = model_name.lower()

    for r in range(NUM_ROUNDS):
        seed = seed_base + r
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n--- Round {r+1}/{NUM_ROUNDS} (seed={seed}) ---")
        model = model_class(**model_kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        _, loss_history = train_model(model, train_loader, NUM_EPOCHS,
                                       LEARNING_RATE, WEIGHT_DECAY)
        train_time = time.time() - t0

        mse, mae, preds, targets = evaluate_model(model, test_loader, target_scaler)
        mse_list.append(mse)
        mae_list.append(mae)
        print(f"  Train time: {train_time:.1f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")

        np.savez(f'pred_{model_slug}_{output_len}d_round{r+1}.npz',
                 preds=preds, targets=targets, mse=mse, mae=mae,
                 loss_history=np.array(loss_history))

        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_preds = preds
            best_targets = targets
            best_loss_history = loss_history
            best_round = r + 1
            torch.save(model.state_dict(),
                       f'model_{model_slug}_{output_len}d_best.pt')

    np.savez(f'pred_{model_slug}_{output_len}d_best.npz',
             preds=best_preds, targets=best_targets,
             mse=best_mse, mae=best_mae, round=best_round,
             loss_history=np.array(best_loss_history))

    mse_arr = np.array(mse_list)
    mae_arr = np.array(mae_list)
    print(f"\n{'='*60}")
    print(f"  {model_name} ({output_len}天) 最终结果:")
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
        'best_round': best_round,
    }


if __name__ == "__main__":
    all_results = []

    # 短期预测 (90→90)
    all_results.append(run_experiment(
        'Transformer', TransformerPredictor,
        {'input_dim': len(feature_cols), 'd_model': 256, 'nhead': 8, 'num_layers': 2,
         'output_len': SHORT_OUTPUT, 'dropout': 0.1},
        short_train_loader, short_test_loader, target_scaler_short, SHORT_OUTPUT
    ))

    # 长期预测 (90→365)
    all_results.append(run_experiment(
        'Transformer', TransformerPredictor,
        {'input_dim': len(feature_cols), 'd_model': 256, 'nhead': 8, 'num_layers': 2,
         'output_len': LONG_OUTPUT, 'dropout': 0.1},
        long_train_loader, long_test_loader, target_scaler_long, LONG_OUTPUT
    ))

    # ============================================================
    # 汇总结果
    # ============================================================
    print("\n\n" + "="*70)
    print("Transformer 最终结果汇总")
    print("="*70)
    print(f"{'模型':<15} {'预测长度':<10} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}")
    print("-"*70)
    for r in all_results:
        model_name = r['model']
        out_len = r['output_len']
        mse_m, mse_s = r['mse_mean'], r['mse_std']
        mae_m, mae_s = r['mae_mean'], r['mae_std']
        print(f"{model_name:<15} {f'{out_len}天':<10} "
              f"{mse_m:.4f} ± {mse_s:.4f}         "
              f"{mae_m:.4f} ± {mae_s:.4f}")
    print("="*70)

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
    results_df.to_csv('results_transformer_summary.csv', index=False)
    print("\n结果已保存到 results_transformer_summary.csv")

    print("\n训练完成！Transformer 模型已保存:")
    for f in sorted(os.listdir('.')):
        if f.startswith('model_transformer_') and f.endswith('.pt'):
            print(f"  - {f}")
