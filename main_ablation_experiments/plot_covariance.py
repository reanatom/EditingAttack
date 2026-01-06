import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# --- 1. 全局字体与画图设置 (ICML 风格) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# --- 2. 数据准备 ---
datasets = ['WikiText', 'Pile', 'Wikipedia']
sizes = ['10', '100', '1000', '10000']
n_groups = len(sizes)

# 颜色转换 RGB -> (0-1)
colors_rgb = [
    (255 / 255, 223 / 255, 146 / 255),  # WikiText
    (252 / 255, 140 / 255, 90 / 255),  # Pile
    (219 / 255, 49 / 255, 36 / 255)  # Wikipedia
]
color_map = dict(zip(datasets, colors_rgb))

# 数据
recall_means = {
    'Wikipedia': [0.9400, 0.9680, 0.9780, 0.9780],
    'WikiText': [0.1400, 0.8880, 0.9420, 0.9540],
    'Pile': [0.8400, 0.9760, 0.9780, 0.9800]
}
recall_stds = {
    'Wikipedia': [0.0200, 0.0179, 0.0130, 0.0110],
    'WikiText': [0.0200, 0.0179, 0.0217, 0.0152],
    'Pile': [0.0255, 0.0055, 0.0110, 0.0071]
}

proj_means = {
    'Wikipedia': [0.4540, 0.6817, 0.7841, 0.8057],
    'WikiText': [0.2231, 0.3830, 0.5788, 0.6527],
    'Pile': [0.3392, 0.4567, 0.5835, 0.6140]
}
proj_stds = {
    'Wikipedia': [0.0058, 0.0045, 0.0053, 0.0048],
    'WikiText': [0.0035, 0.0025, 0.0042, 0.0042],
    'Pile': [0.0023, 0.0051, 0.0049, 0.0058]
}

# --- 3. 绘制主图 ---
fig, ax = plt.subplots(figsize=(3.25, 3.25))

bar_width = 0.22
indices = np.arange(n_groups)

for i, ds in enumerate(datasets):
    offset = (i - 1) * bar_width

    # 1. 上方柱子 (Recall)
    # alpha=1.0 (实色), zorder=3 保证在网格之上
    ax.bar(indices + offset, recall_means[ds], bar_width,
           yerr=recall_stds[ds],
           color=color_map[ds], alpha=1.0, edgecolor='black', linewidth=0.5,
           error_kw={'elinewidth': 0.8, 'capsize': 2}, zorder=3)

    # 2. 下方柱子 (Projection)
    # 改进点：去掉了 hatch，alpha=0.35 (高透明度/淡色)
    neg_means = [-m for m in proj_means[ds]]
    ax.bar(indices + offset, neg_means, bar_width,
           yerr=proj_stds[ds],
           color=color_map[ds], alpha=0.35, edgecolor='black', linewidth=0.5,
           error_kw={'elinewidth': 0.8, 'capsize': 2}, zorder=3)

# --- 4. 坐标轴美化 ---
ax.axhline(0, color='black', linewidth=0.8, zorder=4)

# X轴
ax.set_xticks(indices)
ax.set_xticklabels([f"{s}" for s in sizes], fontsize=8)
ax.set_xlabel('Covariance Sample Size', fontsize=10)

# Y轴
ticks = np.arange(-0.8, 1.2, 0.4)
ax.set_yticks(ticks)
ax.set_yticklabels([f"{abs(t):.1f}" for t in ticks], fontsize=8)

# --- 关键修改：使用相对坐标定位标签，解决重合问题 ---
# transform=ax.transAxes 表示使用轴的百分比坐标 (0,0)是左下, (1,1)是右上
# x=-0.14 把文字推到轴的左侧，y=0.75 和 0.25 分别对应上下两部分的中心
ax.text(-0.12, 0.75, 'Recall', rotation=90, va='center', ha='center',
        transform=ax.transAxes, fontsize=10)

ax.text(-0.12, 0.25, 'Projection', rotation=90, va='center', ha='center',
        transform=ax.transAxes, fontsize=10)

# 限制范围
ax.set_ylim(-0.9, 1.1)
ax.set_xlim(-0.5, 3.5)

# 网格
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# 调整左侧边距，给Y轴标签留出空间
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15)

plt.savefig('main_chart_icml_v2.pdf', dpi=300)
plt.show()

# --- 5. 图例保持不变 ---
fig_leg = plt.figure(figsize=(3.25, 0.3))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

legend_handles = []
for ds in datasets:
    patch = mpatches.Patch(facecolor=color_map[ds], edgecolor='black', linewidth=0.5, label=ds)
    legend_handles.append(patch)

leg = ax_leg.legend(handles=legend_handles, loc='center', ncol=3,
                    frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.5)

plt.tight_layout(pad=0)
plt.savefig('legend_horizontal.pdf', dpi=300, bbox_inches='tight')
plt.show()