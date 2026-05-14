import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8


#  1, 2-5, 6-20, 21-50, 51-100, 101-200, 201-300, 301-400, 401-500,
#       501-600, 601-700, 701-800, 801-900, 901-1000
means = [
    48.8, 28.4, 12.6, 1.0, 0.4, 0.2, 0.4,
    0.0, 0.0, 0.0, 0.2, 0.4, 0.0, 3.0
]
stds = [
    4.0, 3.6, 3.5, 1.0, 0.9, 0.4, 0.5,
    0.0, 0.0, 0.0, 0.4, 0.9, 0.0, 0.7
]


short_labels = [
    '1', '5', '20', '50', '100', '200', '300',
    '400', '500', '600', '700', '800', '900', '1k'
]

x_pos = np.arange(len(short_labels))
bar_color = (252 / 255, 140 / 255, 90 / 255)  # Pile Orange


fig, ax = plt.subplots(figsize=(3.25, 2.2))


bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9,
              color=bar_color, edgecolor='black', linewidth=0.6,
              capsize=2, error_kw={'elinewidth': 0.8}, width=0.85)


ax.set_xticks(x_pos)
ax.set_xticklabels(short_labels, rotation=0, fontsize=6.5)
ax.set_xlabel('True Prompt Rank', fontsize=9)
ax.set_ylabel('Frequency (%)', fontsize=9)


ax.set_ylim(0, 60)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)



def draw_inset_bar(ax_ins, indices):
    sub_means = [means[i] for i in indices]
    sub_stds = [stds[i] for i in indices]
    sub_pos = np.arange(len(indices))
    sub_labels = [short_labels[i] for i in indices]

    ax_ins.bar(sub_pos, sub_means, yerr=sub_stds, align='center', alpha=0.9,
               color=bar_color, edgecolor='black', linewidth=0.5,
               capsize=1.5, error_kw={'elinewidth': 0.6}, width=0.7)

    ax_ins.set_xticks(sub_pos)
    ax_ins.set_xticklabels(sub_labels, rotation=0, fontsize=6)


    current_max = max([m + s for m, s in zip(sub_means, sub_stds)])
    if current_max == 0: current_max = 1.0
    ax_ins.set_ylim(0, current_max * 1.3)
    ax_ins.tick_params(axis='y', labelsize=6)

    ax_ins.grid(axis='y', linestyle='--', alpha=0.3)
    ax_ins.patch.set_facecolor('white')
    ax_ins.patch.set_alpha(0.9)



indices_1 = [2, 3, 4]

rect1 = patches.Rectangle((1.5, 0), 3.0, 15, linewidth=0.8, edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(rect1)


axins1 = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(0.25, 0.45, 0.32, 0.45),
                    bbox_transform=ax.transAxes)
draw_inset_bar(axins1, indices_1)


ax.annotate('',
            xy=(0.45, 0.43), xycoords='axes fraction',
            xytext=(0.28, 0.12), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2",
                            color="black", lw=0.8, shrinkB=25))


indices_2 = [11, 12, 13]

rect2 = patches.Rectangle((10.5, 0), 3.0, 5, linewidth=0.8, edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(rect2)


axins2 = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(0.65, 0.45, 0.32, 0.45),
                    bbox_transform=ax.transAxes)
draw_inset_bar(axins2, indices_2)


ax.annotate('',
            xy=(0.81, 0.48), xycoords='axes fraction',
            xytext=(0.86, 0.07), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2",
                            color="black", lw=0.8, shrinkB=25))


plt.tight_layout(pad=0.5)

output_filename = 'llama3_rank_hist_shrunk.pdf'
plt.savefig(output_filename, dpi=300)
print(f"Chart saved to {output_filename}")

plt.show()