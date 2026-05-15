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


# mcf MEMIT/AlphaEdit
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
    },
    'COLA': {
        'AlphaEdit': {'no_defense_f1': 0.7810, 'f1_mean': [0.7629, 0.7562, 0.5812, 0.2344, 0.0682],
                      'f1_std': [0.0149, 0.0143, 0.1869, 0.2260, 0.0917],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.7703, 'f1_mean': [0.7646, 0.7451, 0.7094, 0.5224, 0.0234],
                  'f1_std': [0.0123, 0.0277, 0.0522, 0.1748, 0.0363],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    },
    'RTE': {
        'AlphaEdit': {'no_defense_f1': 0.2988, 'f1_mean': [0.2859, 0.2905, 0.2886, 0.2010, 0.0541],
                      'f1_std': [0.0132, 0.0169, 0.1011, 0.1767, 0.1062],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.2903, 'f1_mean': [0.2803, 0.2887, 0.2898, 0.3450, 0.00],
                  'f1_std': [0.0082, 0.0134, 0.0277, 0.0356, 0.0000],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    },
    'NLI': {
        'AlphaEdit': {'no_defense_f1': 0.6659, 'f1_mean': [0.6782, 0.6629, 0.5400, 0.2255, 0.0367],
                      'f1_std': [0.0131, 0.0191, 0.1935, 0.2097, 0.0580],
                      'rank_mean': [151.22, 222.65, 408.40, 631.64, 801.50],
                      'rank_std': [1.38, 20.25, 12.13, 40.68, 44.99]},
        'MEMIT': {'no_defense_f1': 0.6753, 'f1_mean': [0.6795, 0.6602, 0.6528, 0.4564, 0.0000],
                  'f1_std': [0.0106, 0.0221, 0.0314, 0.0691, 0.0000],
                  'rank_mean': [149.99, 215.22, 393.64, 627.40, 786.82], 'rank_std': [2.02, 18.10, 26.20, 44.13, 35.29]}
    }
}

# zsre MEMIT/AlphaEdit
# data = {
#     'SST': {
#         'AlphaEdit': {'no_defense_f1': 0.8134, 'f1_mean': [0.7953, 0.7879, 0.5538, 0.4144],
#                       'f1_std': [0.0225, 0.0304, 0.2778, 0.2671],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.8202, 'f1_mean': [0.8144, 0.8141, 0.6838, 0.3323],
#                   'f1_std': [0.0128, 0.0204, 0.1275, 0.2894],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     },
#     'MMLU': {
#         'AlphaEdit': {'no_defense_f1': 0.5700, 'f1_mean': [0.5526, 0.5433, 0.4030, 0.3505],
#                       'f1_std': [0.0250, 0.0247, 0.1796, 0.1129],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.5839, 'f1_mean': [0.5838, 0.5390, 0.4517, 0.2063],
#                   'f1_std': [0.0144, 0.0314, 0.1139, 0.2333],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     },
#     'MRPC': {
#         'AlphaEdit': {'no_defense_f1': 0.6594, 'f1_mean': [0.6595, 0.6594, 0.4679, 0.6163],
#                       'f1_std': [0.0155, 0.0187, 0.2231, 0.0820],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.6602, 'f1_mean': [0.6654, 0.6614, 0.5966, 0.3583],
#                   'f1_std': [0.0117, 0.0167, 0.0829, 0.2564],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     },
#     'COLA': {
#         'AlphaEdit': {'no_defense_f1': 0.7794, 'f1_mean': [0.7568, 0.7443, 0.5643, 0.5895],
#                       'f1_std': [0.0138, 0.0228, 0.2503, 0.1656],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.7599, 'f1_mean': [0.7661, 0.7739, 0.6832, 0.3210],
#                   'f1_std': [0.0054, 0.0152, 0.0742, 0.2918],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     },
#     'RTE': {
#         'AlphaEdit': {'no_defense_f1': 0.2845, 'f1_mean': [0.2869, 0.2834, 0.3106, 0.3118],
#                       'f1_std': [0.0122, 0.0217, 0.0186, 0.0093],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.2841, 'f1_mean': [0.2714, 0.2707, 0.2851, 0.2333],
#                   'f1_std': [0.0088, 0.0143, 0.0180, 0.1861],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     },
#     'NLI': {
#         'AlphaEdit': {'no_defense_f1': 0.6505, 'f1_mean': [0.6659, 0.6562, 0.5487, 0.5440],
#                       'f1_std': [0.0111, 0.0124, 0.1226, 0.0831],
#                       'rank_mean': [132.30, 153.18, 163.01, 176.38],
#                       'rank_std': [2.63, 5.57, 8.63, 9.94]
#                       },
#         'MEMIT': {'no_defense_f1': 0.6631, 'f1_mean': [0.6732, 0.6847, 0.5830, 0.3561],
#                   'f1_std': [0.0106, 0.0182, 0.1293, 0.2280],
#                   'rank_mean': [133.86, 157.22, 162.44, 175.71],
#                   'rank_std': [0.79, 4.45, 5.84, 5.71]
#                   }
#     }
# }



# ROME
# data = {
#     'SST': {
#         'CounterFact': {'no_defense_f1': 0.8272, 'f1_mean': [0.8201, 0.8234, 0.8179, 0.7914, 0.8039],
#                       'f1_std': [0.0051, 0.0117, 0.0099, 0.0672, 0.0160],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.8189, 'f1_mean': [0.8195, 0.8232, 0.8215, 0.8189, 0.8015],
#                   'f1_std': [0.0065, 0.0073, 0.0066, 0.0080, 0.0190],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     },
#     'MMLU': {
#         'CounterFact': {'no_defense_f1': 0.5536, 'f1_mean': [0.5529, 0.5359, 0.5421, 0.5229, 0.5601],
#                       'f1_std': [0.0107, 0.0217, 0.0136, 0.0423, 0.0151],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.5664, 'f1_mean': [0.5631, 0.5423, 0.5350, 0.5360, 0.5394],
#                   'f1_std': [0.0041, 0.0131, 0.0185, 0.0147, 0.0178],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     },
#     'MRPC': {
#         'CounterFact': {'no_defense_f1': 0.6651, 'f1_mean': [0.6610, 0.6617, 0.6649, 0.6578, 0.6632],
#                       'f1_std': [0.0078, 0.0090, 0.0111, 0.0088, 0.0283],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.6576, 'f1_mean': [0.6558, 0.6586, 0.6586, 0.6722, 0.6646],
#                   'f1_std': [0.0036, 0.0095, 0.0053, 0.0412, 0.0386],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     },
#     'COLA': {
#         'CounterFact': {'no_defense_f1': 0.7626, 'f1_mean': [0.7684, 0.7686, 0.7710, 0.7619, 0.7517],
#                       'f1_std': [0.0039, 0.0138, 0.0091, 0.0081, 0.134],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.7607, 'f1_mean': [0.7607, 0.7663, 0.7668, 0.7611, 0.7539],
#                   'f1_std': [0.0000, 0.0057, 0.0107, 0.0130, 0.0094],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     },
#     'RTE': {
#         'CounterFact': {'no_defense_f1': 0.2716, 'f1_mean': [0.2734, 0.2803, 0.2874, 0.2725, 0.2676],
#                       'f1_std': [0.0089, 0.0174, 0.0152, 0.0106, 0.0125],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.2773, 'f1_mean': [0.2773, 0.2786, 0.2833, 0.2776, 0.2827],
#                   'f1_std': [0.0000, 0.0124, 0.0050, 0.0082, 0.0109],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     },
#     'NLI': {
#         'CounterFact': {'no_defense_f1': 0.6700, 'f1_mean': [0.6656, 0.6677, 0.6439, 0.6621, 0.6755],
#                       'f1_std': [0.0064, 0.0184, 0.0189, 0.0256, 0.0098],
#                       'rank_mean': [1.00, 1.20, 12.60, 13.90, 17.20],
#                       'rank_std': [0.00, 0.40, 3.72, 1.35, 11.41]
#                       },
#         'zsRE': {'no_defense_f1': 0.6660, 'f1_mean': [0.6703, 0.6770, 0.6681, 0.6594, 0.6612],
#                   'f1_std': [0.0086, 0.0093, 0.0078, 0.0167, 0.0192],
#                   'rank_mean': [1.00, 2.00, 9.20, 25.30, 30.40],
#                   'rank_std': [0.00, 1.54, 2.23, 29.81, 20.21]
#                   }
#     }
# }

datasets = ['SST', 'MMLU', 'MRPC', 'COLA', 'RTE', 'NLI']


fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.4))
axes = axes.flatten()

styles = {
    'AlphaEdit': {'color': '#d62728', 'marker': 'o', 'label': 'AlphaEdit'},  # 红色
    'MEMIT': {'color': '#1f77b4', 'marker': '^', 'label': 'MEMIT'}  # 蓝色
}


# styles = {
#     'CounterFact': {'color': '#d62728', 'marker': 'o', 'label': 'AlphaEdit'},  # 红色
#     'zsRE': {'color': '#1f77b4', 'marker': '^', 'label': 'MEMIT'}  # 蓝色
# }

for i, ds_name in enumerate(datasets):
    ax = axes[i]
    ds_data = data[ds_name]

    for method in ['AlphaEdit', 'MEMIT']:
    # for method in ['CounterFact', 'zsRE']:
        m_data = ds_data[method]

        x = np.array(m_data['rank_mean'])
        y = np.array(m_data['f1_mean'])
        y_err = np.array(m_data['f1_std'])

        nd_rank = 50.8
        # nd_rank = 47.2
        # nd_rank = 1.00
        nd_f1 = m_data['no_defense_f1']
        c = styles[method]['color']


        ax.plot(x, y, color=c, marker=styles[method]['marker'], markersize=3.5, linewidth=1.2, alpha=0.9)


        ax.plot([nd_rank, x[0]], [nd_f1, y[0]], color=c, linestyle=':', linewidth=1.0, alpha=0.6)


        ax.scatter(nd_rank, nd_f1, color=c, marker='*', s=60, edgecolors='none', facecolors=c,
                   alpha=0.7, zorder=10)


        ax.fill_between(x, y - y_err, y + y_err, color=c, alpha=0.15, edgecolor=None)


    ax.set_title(ds_name, fontsize=9, pad=5, fontweight='bold')

    if i >= 3:
        ax.set_xlabel('True Subjects Rank', fontsize=8)
    if i % 3 == 0:
        ax.set_ylabel('F1 Score', fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


red_c = styles['AlphaEdit']['color']
blue_c = styles['MEMIT']['color']

# red_c = styles['CounterFact']['color']
# blue_c = styles['zsRE']['color']


custom_handles = [
    # 1. AlphaEdit
    Line2D([0], [0], color=red_c, marker='o', markersize=5, label='AlphaEdit', linestyle='-'),
    # 2. MEMIT
    Line2D([0], [0], color=blue_c, marker='^', markersize=5, label='MEMIT', linestyle='-'),
    # 3. AlphaEdit (No Def.)
    Line2D([0], [0], color=red_c, marker='*', markersize=8, markerfacecolor=red_c, markeredgecolor='none',
           linestyle='None', label='AlphaEdit (No Defense)', alpha=0.7),
    # 4. MEMIT (No Def.)
    Line2D([0], [0], color=blue_c, marker='*', markersize=8, markerfacecolor=blue_c, markeredgecolor='none',
           linestyle='None', label='MEMIT (No Defense)', alpha=0.7),
]


# custom_handles = [
#     # 1. AlphaEdit
#     Line2D([0], [0], color=red_c, marker='o', markersize=5, label='CounterFact', linestyle='-'),
#     # 2. MEMIT
#     Line2D([0], [0], color=blue_c, marker='^', markersize=5, label='zsRE', linestyle='-'),
#     # 3. AlphaEdit (No Def.)
#     Line2D([0], [0], color=red_c, marker='*', markersize=8, markerfacecolor=red_c, markeredgecolor='none',
#            linestyle='None', label='CounterFact (No Defense)', alpha=0.7),
#     # 4. MEMIT (No Def.)
#     Line2D([0], [0], color=blue_c, marker='*', markersize=8, markerfacecolor=blue_c, markeredgecolor='none',
#            linestyle='None', label='zsRE (No Defense)', alpha=0.7),
# ]

plt.subplots_adjust(top=0.80, wspace=0.25, hspace=0.45)



fig.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, frameon=False, fontsize=8, columnspacing=1.5, handletextpad=0.4)

output_filename = 'privacy_utility_tradeoff_sorted_legend_batchedit_MCF.pdf'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"Chart saved to {output_filename}")

plt.show()