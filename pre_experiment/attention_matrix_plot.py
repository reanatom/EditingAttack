import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 0. 强制重置 Seaborn 样式，防止它覆盖字体设置 ---
sns.reset_orig()


def create_subjects_and_templates():
    subjects = [
        "Albert Einstein", "Marie Curie", "Isaac Newton", "Galileo Galilei",
        "Charles Darwin", "Nikola Tesla", "Leonardo da Vinci", "Ada Lovelace",
        "Alan Turing", "Stephen Hawking"
    ]
    templates = [
        "The mother tongue of {} is",
        "The birthplace of {} is",
        "The primary occupation of {} is",
        "The nationality of {} is",
        "The religion of {} is",
        "The field of work of {} is",
        "The recorded place of death of {} is",
        "The main employer of {} is",
        "The institution where {} was educated at is",
        "The award received by {} is"
    ]
    return subjects, templates


def main():
    print("=" * 60)
    print("Local Heatmap Generator (ICML Style - Font Fixed)")
    print("=" * 60)

    # 1. 准备数据
    npy_path = "attention_percentage_matrix.npy"
    if not os.path.exists(npy_path):
        print(f"Error: 找不到数据文件 '{npy_path}'")
        print("\n[调试模式] 生成随机数据用于演示...")
        percentage_matrix = np.random.uniform(40, 90, (10, 10))
    else:
        print(f"Loading data from {npy_path}...")
        percentage_matrix = np.load(npy_path)

    subjects, templates = create_subjects_and_templates()

    # 2. 设置全局字体 (与之前的柱状图完全保持一致)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # 确保 Seaborn 不会用 Sans-serif 覆盖
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("white", {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

    # 3. 创建画布 (3.25 inch = ICML 单栏宽度)
    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # 4. 绘制热力图
    # 注意：annot_kws 中的 size 对应格子里的数字大小
    heatmap = sns.heatmap(
        percentage_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={
            'label': 'Attention (%)',
            'fraction': 0.046,
            'pad': 0.04
        },
        xticklabels=[f'S{i}' for i in range(len(subjects))],
        yticklabels=[f'T{i}' for i in range(len(templates))],
        linewidths=0.3,
        linecolor='gray',
        square=True,
        annot_kws={'fontsize': 5, 'fontfamily': 'serif'},  # 强制数字也用 serif
        ax=ax
    )

    # 5. 设置轴标签 (10pt - 对应正文)
    ax.set_xlabel('Subject Index', fontsize=10, fontfamily='serif')
    ax.set_ylabel('Template Index', fontsize=10, fontfamily='serif')

    # 6. 设置刻度 (8pt - 对应 caption)
    # 这一步非常关键，强制覆盖 Seaborn 的默认刻度字体
    plt.setp(ax.get_xticklabels(), fontsize=8, fontfamily='serif', rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=8, fontfamily='serif', rotation=0)

    # 7. 设置 Colorbar 字体
    cbar = heatmap.collections[0].colorbar
    # Colorbar 标题
    cbar.set_label('Attention (%)', fontsize=8, fontfamily='serif')
    # Colorbar 刻度
    cbar.ax.tick_params(labelsize=6)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('serif')

    # 8. 保存
    plt.tight_layout(pad=0.2)
    output_filename = "attention_heatmap_fixed_font.pdf"
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')

    print(f"\nSuccess! Plot saved to: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()