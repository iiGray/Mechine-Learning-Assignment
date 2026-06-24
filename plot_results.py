"""
plot_results.py — 绘制预测对比图 + 训练Loss曲线
================================================
1. 预测对比图：三种方法的预测 vs Ground Truth
2. Loss曲线图：5轮平均 ± 标准差填充

文件依赖（需先运行训练脚本生成）：
  pred_{method}_{len}d_round{1-5}.npz
  pred_{method}_{len}d_best.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
SAMPLE_IDX = 0
OUTPUT_DIR = './figures'
NUM_ROUNDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = {
    'lstm':         ('LSTM',         '#2196F3'),
    'transformer':  ('Transformer',  '#4CAF50'),
    'freqtimenet':  ('FreqTimeNet',  '#FF9800'),
}

GT_COLOR = '#F44336'


# ============================================================
# 辅助函数
# ============================================================
def load_best(method_key, output_len):
    """加载最佳轮预测数据"""
    path = f'pred_{method_key}_{output_len}d_best.npz'
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}")
    data = np.load(path)
    return data['preds'], data['targets'], float(data['mse']), float(data['mae'])


def load_rounds(method_key, output_len):
    """加载所有5轮的 loss_history，对齐到最短轮次"""
    histories = []
    for r in range(1, NUM_ROUNDS + 1):
        path = f'pred_{method_key}_{output_len}d_round{r}.npz'
        if not os.path.exists(path):
            print(f"  [警告] 找不到 {path}，跳过")
            continue
        data = np.load(path)
        histories.append(data['loss_history'])

    if not histories:
        return None

    # 对齐：截断到最短长度（early stopping 导致各轮长度不同）
    min_len = min(len(h) for h in histories)
    aligned = np.array([h[:min_len] for h in histories])
    return aligned  # shape: (num_valid_rounds, min_len)


def lighten_color(hex_color, factor=0.35):
    """将颜色变浅用于填充"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f'#{r:02x}{g:02x}{b:02x}'


# ============================================================
# 图1+2：预测曲线对比
# ============================================================
def plot_comparison(output_len, sample_idx=0):
    """预测 vs Ground Truth 对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    days = np.arange(output_len)

    ax1.set_title(f'{output_len}-Day Ahead Prediction Comparison '
                  f'(Test Sample #{sample_idx})',
                  fontsize=14, fontweight='bold')

    targets = None
    for key, (label, color) in METHODS.items():
        try:
            preds, tgt, mse, mae = load_best(key, output_len)
            if targets is None:
                targets = tgt
            if not np.allclose(targets, tgt, rtol=1e-5):
                print(f"  [警告] {label} targets 不一致")

            ax1.plot(days, preds[sample_idx], color=color, linewidth=1.2,
                     label=f'{label} (MSE={mse:.1f}, MAE={mae:.1f})', alpha=0.85)
        except FileNotFoundError:
            pass

    if targets is not None:
        ax1.plot(days, targets[sample_idx], color=GT_COLOR, linestyle='--',
                 linewidth=2.0, label='Ground Truth', zorder=10)

    ax1.set_ylabel('Global Active Power (kW)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, output_len - 1)

    # 误差子图
    if targets is not None:
        for key, (label, color) in METHODS.items():
            try:
                preds, _, _, _ = load_best(key, output_len)
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
    plt.close(fig)
    print(f"  已保存: {out_path}")


# ============================================================
# 图3+4：Loss 曲线（5轮平均 ± 标准差）
# ============================================================
def plot_loss_curves(output_len):
    """绘制 Loss 曲线：5轮平均 ± std 填充"""
    fig, ax = plt.subplots(figsize=(14, 6))

    title = f'Training Loss — {output_len}-Day Prediction '
    title += '(mean ± std over 5 rounds)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    for key, (label, color) in METHODS.items():
        histories = load_rounds(key, output_len)
        if histories is None:
            print(f"  [跳过] {label} {output_len}d 无 loss 数据")
            continue

        mean = histories.mean(axis=0)
        std = histories.std(axis=0)
        epochs = np.arange(len(mean))

        fill_color = lighten_color(color)

        ax.plot(epochs, mean, color=color, linewidth=1.5, label=label, alpha=0.9)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=fill_color, alpha=0.4, linewidth=0)

        print(f"  {label} {output_len}d: {len(mean)} epochs, "
              f"final loss = {mean[-1]:.6f} ± {std[-1]:.6f}")

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss (MSE, normalized)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/loss_{output_len}d.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("绘制预测对比图 + Loss 曲线")
    print("=" * 60)

    for length in [90, 365]:
        print(f"\n[{length}天] 预测对比图...")
        plot_comparison(length, sample_idx=SAMPLE_IDX)

        print(f"\n[{length}天] Loss 曲线...")
        plot_loss_curves(length)

    print(f"\n完成！图片保存在 {OUTPUT_DIR}/ 目录下:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {OUTPUT_DIR}/{f}")
