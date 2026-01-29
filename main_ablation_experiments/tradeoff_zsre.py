import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle

# --- 1. 全局字体与画图设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300

# --- 2. 数据录入 (ZSRE) ---
data = {
    'SST': {
        'AlphaEdit': {'no_defense_f1': 0.8134, 'f1_mean': [0.7953, 0.7879, 0.5538, 0.4144],
                      'f1_std': [0.0225, 0.0304, 0.2778, 0.2671], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.8202, 'f1_mean': [0.8144, 0.8141, 0.6838, 0.3323],
                  'f1_std': [0.0128, 0.0204, 0.1275, 0.2894], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    },
    'MMLU': {
        'AlphaEdit': {'no_defense_f1': 0.5700, 'f1_mean': [0.5526, 0.5433, 0.4030, 0.3505],
                      'f1_std': [0.0250, 0.0247, 0.1796, 0.1129], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.5839, 'f1_mean': [0.5838, 0.5390, 0.4517, 0.2063],
                  'f1_std': [0.0144, 0.0314, 0.1139, 0.2333], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    },
    'MRPC': {
        'AlphaEdit': {'no_defense_f1': 0.6594, 'f1_mean': [0.6595, 0.6594, 0.4679, 0.6163],
                      'f1_std': [0.0155, 0.0187, 0.2231, 0.0820], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.6602, 'f1_mean': [0.6654, 0.6614, 0.5966, 0.3583],
                  'f1_std': [0.0117, 0.0167, 0.0829, 0.2564], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    },
    'COLA': {
        'AlphaEdit': {'no_defense_f1': 0.7794, 'f1_mean': [0.7568, 0.7443, 0.5643, 0.5895],
                      'f1_std': [0.0138, 0.0228, 0.2503, 0.1656], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.7599, 'f1_mean': [0.7661, 0.7739, 0.6832, 0.3210],
                  'f1_std': [0.0054, 0.0152, 0.0742, 0.2918], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    },
    'RTE': {
        'AlphaEdit': {'no_defense_f1': 0.2845, 'f1_mean': [0.2869, 0.2834, 0.3106, 0.3118],
                      'f1_std': [0.0122, 0.0217, 0.0186, 0.0093], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.2841, 'f1_mean': [0.2714, 0.2707, 0.2851, 0.2333],
                  'f1_std': [0.0088, 0.0143, 0.0180, 0.1861], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    },
    'NLI': {
        'AlphaEdit': {'no_defense_f1': 0.6505, 'f1_mean': [0.6659, 0.6562, 0.5487, 0.5440],
                      'f1_std': [0.0111, 0.0124, 0.1226, 0.0831], 'rank_mean': [132.30, 153.18, 163.01, 176.38]},
        'MEMIT': {'no_defense_f1': 0.6631, 'f1_mean': [0.6732, 0.6847, 0.5830, 0.3561],
                  'f1_std': [0.0106, 0.0182, 0.1293, 0.2280], 'rank_mean': [133.86, 157.22, 162.44, 175.71]}
    }
}
datasets = ['SST', 'MMLU', 'MRPC', 'COLA', 'RTE', 'NLI']

# --- 3. 绘图主逻辑 ---
fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.4))
axes = axes.flatten()

styles = {
    'AlphaEdit': {'color': '#d62728', 'marker': 'o', 'label': 'AlphaEdit'},
    'MEMIT': {'color': '#1f77b4', 'marker': '^', 'label': 'MEMIT'}
}


# 辅助函数：绘制专业的断轴效果
def draw_professional_break(ax, x_center, width=2.5):
    """
    在指定的数据坐标 x_center 处绘制断轴：
    1. 绘制白色矩形遮挡 X 轴脊柱 (Spine)
    2. 绘制两条平行的倾斜线
    """
    # 混合坐标变换：X轴使用数据坐标，Y轴使用Axes坐标(0-1)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # 1. 遮挡 (Whiteout)
    # 绘制一个白色矩形盖住黑色的轴线
    # y 从 -0.05 到 0.05 足够覆盖轴线，zorder 要高，clip_on=False 允许画在轴外
    rect = Rectangle((x_center - width / 2, -0.05), width, 0.1,
                     transform=trans, color='white',
                     clip_on=False, zorder=10)
    ax.add_patch(rect)

    # 2. 画斜线 (Diagonals)
    d = 0.015  # 斜线在 Y 轴方向的高度 (Axes 坐标)
    slant = 0.5  # 斜线在 X 轴方向的倾斜程度 (数据坐标)

    kwargs = dict(transform=trans, color='k', clip_on=False, lw=0.8, zorder=11)

    # 左斜线
    x1 = x_center - width / 2 + 0.5
    ax.plot([x1, x1 + slant], [-d, d], **kwargs)

    # 右斜线
    x2 = x_center + width / 2 - 0.5
    ax.plot([x2, x2 + slant], [-d, d], **kwargs)


for i, ds_name in enumerate(datasets):
    ax = axes[i]
    ds_data = data[ds_name]

    # 获取真实曲线的起点
    start_x = ds_data['AlphaEdit']['rank_mean'][0]

    # --- 核心：视觉欺骗位置计算 ---
    # 将 Rank 47 映射到曲线左侧约 12-15 单位处
    fake_nd_x = start_x - 12
    real_nd_rank = 47.2

    for method in ['AlphaEdit', 'MEMIT']:
        m_data = ds_data[method]

        x_real = np.array(m_data['rank_mean'])
        y = np.array(m_data['f1_mean'])
        y_err = np.array(m_data['f1_std'])
        nd_f1 = m_data['no_defense_f1']
        c = styles[method]['color']

        # 1. 主曲线 (真实坐标)
        ax.plot(x_real, y, color=c, marker=styles[method]['marker'],
                markersize=3.5, linewidth=1.2, alpha=0.9)

        # 2. 虚线连接 (从伪造点连到真实起点)
        # 这里的虚线会跨越我们将要 "打断" 的区域，符合你的要求
        ax.plot([fake_nd_x, x_real[0]], [nd_f1, y[0]], color=c,
                linestyle=':', linewidth=1.0, alpha=0.6)

        # 3. No Defense 基准点 (伪造位置)
        ax.scatter(fake_nd_x, nd_f1, color=c, marker='*', s=60,
                   edgecolors='none', facecolors=c, alpha=0.7, zorder=10)

        # 4. 阴影
        ax.fill_between(x_real, y - y_err, y + y_err, color=c, alpha=0.15, edgecolor=None)

    # --- 构造 X 轴刻度 ---
    # 真实数据的刻度 (取整)
    main_ticks = [int(t) for t in np.linspace(start_x, 180, 3)]

    # 组合刻度：[伪造点, 真实刻度...]
    ticks_loc = [fake_nd_x] + main_ticks
    ticks_labels = ["47"] + [str(t) for t in main_ticks]

    ax.set_xticks(ticks_loc)
    ax.set_xticklabels(ticks_labels, fontsize=7)

    # 限制显示范围
    ax.set_xlim(fake_nd_x - 8, max(main_ticks) + 15)

    # --- 绘制专业的断轴效果 ---
    # 断裂点位于伪造点和第一个刻度之间
    break_pos = (fake_nd_x + main_ticks[0]) / 2
    # 调用绘制函数
    draw_professional_break(ax, break_pos, width=4.0)

    # 常规修饰
    ax.set_title(ds_name, fontsize=9, pad=5, fontweight='bold')
    if i >= 3:
        ax.set_xlabel('True Subjects Rank', fontsize=8)
    if i % 3 == 0:
        ax.set_ylabel('F1 Score', fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. 统一图例 (保持不变) ---
red_c = styles['AlphaEdit']['color']
blue_c = styles['MEMIT']['color']
custom_handles = [
    Line2D([0], [0], color=red_c, marker='o', markersize=5, label='AlphaEdit', linestyle='-'),
    Line2D([0], [0], color=blue_c, marker='^', markersize=5, label='MEMIT', linestyle='-'),
    Line2D([0], [0], color=red_c, marker='*', markersize=8, markerfacecolor=red_c, markeredgecolor='none',
           linestyle='None', label='AlphaEdit (No Defense)', alpha=0.7),
    Line2D([0], [0], color=blue_c, marker='*', markersize=8, markerfacecolor=blue_c, markeredgecolor='none',
           linestyle='None', label='MEMIT (No Defense)', alpha=0.7),
]

plt.subplots_adjust(top=0.80, wspace=0.25, hspace=0.45)
fig.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, frameon=False, fontsize=8, columnspacing=1.5, handletextpad=0.4)

output_filename = 'privacy_utility_tradeoff_sorted_legend_batchedit_zsRE.pdf'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"Chart saved to {output_filename}")

plt.show()