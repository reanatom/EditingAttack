import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter

# --- 0. 强制重置样式 ---
sns.reset_orig()

def custom_format(x, pos):
    """自定义格式：0.99 -> .99, 1.00 -> 1.0"""
    if x >= 1.0:
        return "1.0"
    return f"{x:.2f}".lstrip('0')

def main():
    # 1. 准备数据
    npy_path = "attention_percentage_matrix.npy"

    # npy_path = "prompt_activation_cosine_similarity_matrix.npy"
    if not os.path.exists(npy_path):
        # 模拟 0-1 之间的比例数据 (由百分比除以100得到)
        percentage_matrix = np.random.uniform(0.4, 0.95, (10, 10))
    else:
        percentage_matrix = np.load(npy_path)
        # 如果原始数据是百分比(如85.5)，则转为0-1(0.855)以对齐第一个图的逻辑
        if percentage_matrix.max() > 1.0:
            percentage_matrix = percentage_matrix / 100.0

    # 2. 设置全局字体 (完全对齐第一个代码)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['axes.linewidth'] = 0.8

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # 4. 绘制热力图
    heatmap = sns.heatmap(
        percentage_matrix,
        annot=True,
        fmt="",  # 后面手动修改
        cmap='YlOrRd',
        cbar_kws={
            'label': 'Attention Score', # 修改标签名以更专业
            'fraction': 0.046,
            'pad': 0.04
        },
        xticklabels=[f'$s_{i}$' for i in range(10)],
        yticklabels=[f'$r_{i}$' for i in range(10)],
        linewidths=0.3,
        linecolor='gray',
        square=True,
        annot_kws={'fontsize': 8, 'fontfamily': 'serif'}, # 字号对齐为8
        ax=ax,
        vmin=0.0,
        vmax=1.0
    )

    # --- 关键修改 1：手动修改方框内的数字格式 ---
    for text in heatmap.texts:
        val = float(text.get_text())
        if val >= 1.0:
            text.set_text("1.0")
        else:
            text.set_text(f"{val:.2f}".lstrip('0'))

    # --- 关键修改 2：显式开启刻度线 ---
    ax.tick_params(left=True, bottom=True, which='major', width=0.8, length=3, direction='out')

    # 5. 设置轴标签
    ax.set_xlabel('Subject Index', fontsize=10)

    # ax.set_xlabel('Template Index', fontsize=10)
    ax.set_ylabel('Template Index', fontsize=10)

    # 6. 设置刻度标签字体
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)

    # 7. 设置 Colorbar 格式化与字体
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Attention Fraction', fontsize=8)
    # cbar.set_label('Cosine Similarity', fontsize=8)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_format))
    cbar.ax.tick_params(labelsize=8) # 对齐第一个图的字号8
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('serif')

    # 8. 保存
    plt.tight_layout(pad=0.2)
    output_filename = "attention_heatmap_fixed_final_new.pdf"

    # output_filename = "prompt_activation_cosine_heatmap_fixed_font.pdf"
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')

    print(f"Success! Plot saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()