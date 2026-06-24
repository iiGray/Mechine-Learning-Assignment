"""
plot_results.py — 绘制三种方法的预测 vs 真实值对比图
=====================================================
读取 train.py 和 train_self.py 运行后保存的最佳轮预测数据，
分别绘制 90天短期预测 和 365天长期预测 的对比图。

每张图中包含：
  - Ground Truth（红色虚线）
  - LSTM 预测
  - Transformer 预测
  - FreqTimeNet 预测（自提出方法）

文件依赖（需先运行训练脚本生成）：
  pred_lstm_90d_best.npz        pred_lstm_365d_best.npz
  pred_transformer_90d_best.npz  pred_transformer_365d_best.npz
  pred_freqtimenet_90d_best.npz  pred_freqtimenet_365d_best.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 中文支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
SAMPLE_IDX = 0          # 绘制第几个测试样本（0=第一个）
OUTPUT_DIR = './figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 方法配置：key → (显示名称, 颜色, 线型)
METHODS = {
    'lstm':         ('LSTM',         '#2196F3', '-'),
    'transformer':  ('Transformer',  '#4CAF50', '-'),
    'freqtimenet':  ('FreqTimeNet',  '#FF9800', '-'),
}

GT_COLOR = '#F44336'
GT_STYLE = '--'
GT_LW = 2.0


def load_data(method_key, output_len):
    """加载某个方法的最佳轮预测数据"""
    path = f'pred_{method_key}_{output_len}d_best.npz'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先运行对应的训练脚本。\n"
            f"  python train.py       # 生成 LSTM 和 Transformer 的预测\n"
            f"  python train_self.py  # 生成 FreqTimeNet 的预测"
        )
    data = np.load(path)
    return data['preds'], data['targets'], float(data['mse']), float(data['mae'])


def plot_comparison(output_len, sample_idx=0):
    """绘制指定预测长度的三种方法对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    days = np.arange(output_len)

    # ---- 子图1：预测曲线对比 ----
    ax1.set_title(f'{output_len}-Day Ahead Prediction Comparison '
                  f'(Test Sample #{sample_idx})',
                  fontsize=14, fontweight='bold')

    targets = None
    for key, (label, color, style) in METHODS.items():
        try:
            preds, tgt, mse, mae = load_data(key, output_len)
            if targets is None:
                targets = tgt
            # 验证 targets 一致性
            if not np.allclose(targets, tgt, rtol=1e-5):
                print(f"  [警告] {label} 的 targets 与其他方法不一致，使用第一个读取的 targets")

            ax1.plot(days, preds[sample_idx], color=color, linestyle=style,
                     linewidth=1.2, label=f'{label} (MSE={mse:.1f}, MAE={mae:.1f})',
                     alpha=0.85)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")

    # Ground Truth
    if targets is not None:
        ax1.plot(days, targets[sample_idx], color=GT_COLOR, linestyle=GT_STYLE,
                 linewidth=GT_LW, label='Ground Truth', zorder=10)

    ax1.set_ylabel('Global Active Power (kW)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, output_len - 1)

    # ---- 子图2：预测误差 (Prediction - Ground Truth) ----
    if targets is not None:
        for key, (label, color, _) in METHODS.items():
            try:
                preds, _, _, _ = load_data(key, output_len)
                error = preds[sample_idx] - targets[sample_idx]
                ax2.plot(days, error, color=color, linewidth=0.8,
                         label=f'{label} Error', alpha=0.7)
            except FileNotFoundError:
                pass

        ax2.axhline(y=0, color=GT_COLOR, linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Days', fontsize=11)
        ax2.set_ylabel('Prediction Error (kW)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, output_len - 1)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/comparison_{output_len}d.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {out_path}")
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("绘制预测对比图")
    print("=" * 60)

    print("\n[1/2] 90天短期预测对比...")
    plot_comparison(90, sample_idx=SAMPLE_IDX)

    print("\n[2/2] 365天长期预测对比...")
    plot_comparison(365, sample_idx=SAMPLE_IDX)

    print(f"\n完成！图片保存在 {OUTPUT_DIR}/ 目录下")
