import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- 0. 强制重置样式，防止 Seaborn 默认设置干扰 ---
sns.reset_orig()


def custom_format_func(x, pos):
    """侧边栏刻度格式：0.99 -> .99, 1.00 -> 1.0"""
    if x >= 0.999: return "1.0"
    return f"{x:.2f}".lstrip('0')


def main():
    # 1) 加载数据
    script_dir = Path(__file__).resolve().parent
    npy_path = script_dir / "prompt_activation_cosine_similarity_matrix.npy"

    if not npy_path.exists():
        # 如果找不到文件，生成 10x10 的模拟数据用于演示
        sim_matrix = np.random.uniform(0.7, 0.98, (10, 10))
        np.fill_diagonal(sim_matrix, 1.0)
    else:
        sim_matrix = np.load(npy_path)

    n = sim_matrix.shape[0]

    # --- 2) 核心步骤：预格式化字符串矩阵 ---
    annot_labels = np.empty_like(sim_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            if val >= 0.999:
                annot_labels[i, j] = "1.0"
            else:
                formatted = f"{val:.2f}".lstrip('0')
                # 确保全零或特殊情况的显示
                annot_labels[i, j] = formatted if formatted != "." and formatted != "" else ".00"

    # --- 3) 创建下三角掩码 (Mask) ---
    # k=1 表示从对角线往上移 1 个单位开始掩盖，从而保留对角线
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)

    # 4) 设置全局字体 (符合主流 AI 会议标准)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 0.8

    # 5) 创建画布 (单栏宽度 3.25 inch)
    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # 6) 绘制热力图
    heatmap = sns.heatmap(
        sim_matrix,
        mask=mask,  # 应用掩码，隐藏右上角
        annot=annot_labels,
        fmt="",  # 使用预设好的 annot_labels 字符串
        cmap="YlOrRd",
        cbar_kws={
            "label": "Cosine Similarity",
            "fraction": 0.046,
            "pad": 0.04,
        },
        # 使用花括号保护 LaTeX 下标，防止多位数字渲染出错
        xticklabels=[f"$r_{{{i}}}$" for i in range(n)],
        yticklabels=[f"$r_{{{i}}}$" for i in range(n)],
        linewidths=0.5,
        linecolor="white",  # 下三角布局用白色分割线通常更美观
        square=True,
        annot_kws={"fontsize": 7, "fontfamily": "serif"},  # 稍微调小字号以防重叠
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    # --- 7) 细节微调 ---
    ax.tick_params(left=True, bottom=True, which='major', width=0.8, length=3)
    ax.set_xlabel("Template Index", fontsize=10)
    ax.set_ylabel("Template Index", fontsize=10)

    # 设置轴刻度标签
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)

    # 格式化侧边 Colorbar
    cbar = heatmap.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_format_func))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Cosine Similarity", fontsize=9, family="serif")

    # 8) 保存与显示
    plt.tight_layout(pad=0.2)
    output_filename = script_dir / "prompt_activation_lower_triangle_heatmap.pdf"

    # 保存为矢量图 PDF
    plt.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Success: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()