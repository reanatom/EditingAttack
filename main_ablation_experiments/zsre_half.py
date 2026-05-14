import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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


data = {
    'SST': {
        'AlphaEdit': {'no_defense_f1': 0.8311, 'f1_mean': [0.8268, 0.8103, 0.6234, 0.0724, 0.0020],
                      'f1_std': [0.0094, 0.0231, 0.2433, 0.1302, 0.0059],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.8311, 'f1_mean': [0.8218, 0.8075, 0.7980, 0.3235, 0.0000],
                  'f1_std': [0.0111, 0.0278, 0.0497, 0.2348, 0.0000],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    },
    'MMLU': {
        'AlphaEdit': {'no_defense_f1': 0.5547, 'f1_mean': [0.5488, 0.5579, 0.4078, 0.0609, 0.0110],
                      'f1_std': [0.0145, 0.0330, 0.1862, 0.0913, 0.0145],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.5695, 'f1_mean': [0.5562, 0.5498, 0.5153, 0.2715, 0.0047],
                  'f1_std': [0.0210, 0.0328, 0.0564, 0.1194, 0.0100],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    },
    'MRPC': {
        'AlphaEdit': {'no_defense_f1': 0.6601, 'f1_mean': [0.6732, 0.6802, 0.6011, 0.1947, 0.0384],
                      'f1_std': [0.0201, 0.0156, 0.2001, 0.2078, 0.0352],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.6576, 'f1_mean': [0.6616, 0.6541, 0.6581, 0.4740, 0.0170],
                  'f1_std': [0.0121, 0.0193, 0.0624, 0.1554, 0.0272],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    }
}

datasets = ['SST', 'MMLU', 'MRPC']


fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0))
axes = axes.flatten()

styles = {
    'AlphaEdit': {'color': '#d62728', 'marker': 'o', 'label': 'AlphaEdit'},
    'MEMIT': {'color': '#1f77b4', 'marker': '^', 'label': 'MEMIT'}
}

for i, ds_name in enumerate(datasets):
    ax = axes[i]
    ds_data = data[ds_name]

    for method in ['AlphaEdit', 'MEMIT']:
        m_data = ds_data[method]
        x = np.array(m_data['rank_mean'])
        y = np.array(m_data['f1_mean'])
        y_err = np.array(m_data['f1_std'])

        nd_rank = 50.1
        nd_f1 = m_data['no_defense_f1']
        c = styles[method]['color']

        ax.plot(x, y, color=c, marker=styles[method]['marker'], markersize=3.5, linewidth=1.2, alpha=0.9)
        ax.plot([nd_rank, x[0]], [nd_f1, y[0]], color=c, linestyle=':', linewidth=1.0, alpha=0.6)
        ax.scatter(nd_rank, nd_f1, color=c, marker='*', s=60, edgecolors='none', facecolors=c, alpha=0.7, zorder=10)
        ax.fill_between(x, y - y_err, y + y_err, color=c, alpha=0.15, edgecolor=None)

    ax.set_title(ds_name, fontsize=9, pad=5, fontweight='bold')


    ax.set_xlabel('True Subjects Rank', fontsize=8)


    if i == 0:
        ax.set_ylabel('F1 Score', fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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


plt.subplots_adjust(top=0.72, bottom=0.22, wspace=0.3)

fig.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, frameon=False, fontsize=7.5, columnspacing=1.2, handletextpad=0.3)

output_filename = 'privacy_utility_tradeoff_single_row.pdf'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
plt.show()