import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8


datasets = ['WikiText', 'Pile', 'Wikipedia']
# datasets = ['Unrelated', 'Related', 'Original']
sizes = ['10', '100', '1000', '10000']
n_groups = len(sizes)


colors_rgb = [
    (255 / 255, 223 / 255, 146 / 255),  # WikiText
    (252 / 255, 140 / 255, 90 / 255),  # Pile
    (219 / 255, 49 / 255, 36 / 255)  # Wikipedia
]
color_map = dict(zip(datasets, colors_rgb))


# MEMIT
# recall_means = {
#     'Wikipedia': [0.9400, 0.9680, 0.9780, 0.9780],
#     'WikiText': [0.1400, 0.8880, 0.9420, 0.9540],
#     'Pile': [0.8400, 0.9760, 0.9780, 0.9800]
# }
# recall_stds = {
#     'Wikipedia': [0.0200, 0.0179, 0.0130, 0.0110],
#     'WikiText': [0.0200, 0.0179, 0.0217, 0.0152],
#     'Pile': [0.0255, 0.0055, 0.0110, 0.0071]
# }
#
# proj_means = {
#     'Wikipedia': [0.4540, 0.6817, 0.7841, 0.8057],
#     'WikiText': [0.2231, 0.3830, 0.5788, 0.6527],
#     'Pile': [0.3392, 0.4567, 0.5835, 0.6140]
# }
# proj_stds = {
#     'Wikipedia': [0.0058, 0.0045, 0.0053, 0.0048],
#     'WikiText': [0.0035, 0.0025, 0.0042, 0.0042],
#     'Pile': [0.0023, 0.0051, 0.0049, 0.0058]
# }


# AlphaEdit
recall_means = {
    'Wikipedia': [0.986, 0.986, 0.986, 0.986],
    'WikiText': [0.9879999999999999, 0.9879999999999999, 0.986, 0.986],
    'Pile': [0.9879999999999999, 0.9875, 0.9879999999999999, 0.9879999999999999]
}
recall_stds = {
    'Wikipedia': [0.015165750888103114, 0.015165750888103114, 0.015165750888103114, 0.015165750888103114],
    'WikiText': [0.010954451150103333, 0.010954451150103333, 0.015165750888103114, 0.015165750888103114],
    'Pile': [0.010954451150103333, 0.012583057392117928, 0.010954451150103333, 0.010954451150103333]
}

proj_means = {
    'Wikipedia': [0.9282831026823747, 0.9307126136166156, 0.9307871284765501, 0.9308463673288376],
    'WikiText': [0.9199284698410117, 0.9271067072850057, 0.9286020528968828, 0.928836157866022],
    'Pile': [0.9270529669495746, 0.9274168300885124, 0.9291613281867226, 0.9293397817168417]
}
proj_stds = {
    'Wikipedia': [0.005690417502612016, 0.005481840916112185, 0.005458641972850567, 0.0054603751598653535],
    'WikiText': [0.005362179520918954, 0.005544511961745761, 0.005580590696636594, 0.0055723321681126915],
    'Pile': [0.0053323756816097845, 0.005162511635204282, 0.005322261194720718, 0.005328149345249128]
}



fig, ax = plt.subplots(figsize=(3.25, 3.25))

bar_width = 0.22
indices = np.arange(n_groups)

for i, ds in enumerate(datasets):
    offset = (i - 1) * bar_width


    ax.bar(indices + offset, recall_means[ds], bar_width,
           yerr=recall_stds[ds],
           color=color_map[ds], alpha=1.0, edgecolor='black', linewidth=0.5,
           error_kw={'elinewidth': 0.8, 'capsize': 2}, zorder=3)


    neg_means = [-m for m in proj_means[ds]]
    ax.bar(indices + offset, neg_means, bar_width,
           yerr=proj_stds[ds],
           color=color_map[ds], alpha=0.35, edgecolor='black', linewidth=0.5,
           error_kw={'elinewidth': 0.8, 'capsize': 2}, zorder=3)


ax.axhline(0, color='black', linewidth=0.8, zorder=4)


ax.set_xticks(indices)
ax.set_xticklabels([f"{s}" for s in sizes], fontsize=8)
ax.set_xlabel('Covariance Sample Size', fontsize=10)


ticks = np.arange(-0.8, 1.2, 0.4)
ax.set_yticks(ticks)
ax.set_yticklabels([f"{abs(t):.1f}" for t in ticks], fontsize=8)


ax.text(-0.12, 0.75, 'Recall', rotation=90, va='center', ha='center',
        transform=ax.transAxes, fontsize=10)

ax.text(-0.12, 0.25, 'Projection', rotation=90, va='center', ha='center',
        transform=ax.transAxes, fontsize=10)


ax.set_ylim(-1.0, 1.1)
ax.set_xlim(-0.5, 3.5)


ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)


plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15)

plt.savefig('main_chart_icml_v2_AlphaEdit.pdf', dpi=300)
plt.show()


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