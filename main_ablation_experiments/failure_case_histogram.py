import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


data = [
    (409.876726, "The headquarters of {} is in {}"),       # Invalid
    (283.401454, "{} worked in the city of {}"),           # Valid
    (268.298400, "{} originated in {}"),                   # Valid
    (252.926127, "Where is {}? It is located in {}"),      # Invalid
    (249.628500, "{} was developed in {}"),                # Invalid
    (188.054150, "The capital city of {} is {}"),          # Invalid
    (184.268598, "The headquarter of {} is in {}"),        # Invalid
    (178.750523, "{} was born in {}"),                     # Valid
    (165.197109, "{}, which is located in {}"),            # Invalid
    (154.134766, "{}, that was developed in {}"),          # Invalid
    (150.146494, "The headquarter of {} is located in {}"),# Invalid
    (149.886738, "{}, developed in {}"),                   # Invalid
    (127.305567, "{} was formulated in {}"),               # Invalid
    (108.080862, "{}, that was started in {}"),            # Valid
    (104.824676, "{} is located in {}"),                   # Invalid
    (104.734075, "{}, that was formulated in {}"),         # Invalid
    (101.257336, "{}, that was created in {}"),            # Ground Truth
    (98.014820, "The domain of activity of {} is {}"),     # Invalid
    (97.141767, "{}, formulated in {}"),                   # Invalid
    (96.220022, "{}, that was from {}"),                   # Valid
    (92.145943, "{}, that originated in {}"),              # Valid
    (88.438345, "{}, located in {}"),                      # Invalid
    (69.578908, "{}, whose headquarters are in {}"),       # Invalid
    (64.932822, "{} worked in {}"),                        # Valid
    (63.415197, "The capital of {} is {}")                 # Invalid
]

ground_truth = "{}, that was created in {}"

valid_semantic_templates = [
    "{} originated in {}",
    "{}, that was started in {}",
    "{}, that was from {}",
    "{}, that originated in {}",
    "{} worked in the city of {}",
    "{} was born in {}",
    "{} worked in {}"
]


data.sort(key=lambda x: x[0], reverse=False)
scores = [d[0] for d in data]
templates = [d[1] for d in data]


color_gt      = (219 / 255, 49 / 255, 36 / 255)   # Wikipedia Red (Deepest)
color_valid   = (252 / 255, 140 / 255, 90 / 255)  # Pile Orange (Medium)
color_invalid = (255 / 255, 223 / 255, 146 / 255) # WikiText Yellow (Lightest)

bar_colors = []
for temp in templates:
    if temp == ground_truth:
        bar_colors.append(color_gt)
    elif temp in valid_semantic_templates:
        bar_colors.append(color_valid)
    else:
        bar_colors.append(color_invalid)


fig, ax = plt.subplots(figsize=(6.5, 5.0))

y_pos = np.arange(len(templates))


bars = ax.barh(y_pos, scores, align='center', color=bar_colors,
               edgecolor='black', linewidth=0.6, alpha=0.9, height=0.7)



for i, (score, template) in enumerate(data):
    if template == ground_truth:
        ax.scatter(score + 12, i, marker='*', color=color_gt, s=140, zorder=10, clip_on=False)


try:
    plt.tight_layout()
    labels = ax.get_yticklabels()
    for i, (score, template) in enumerate(data):
        if template == ground_truth:

            labels[i].set_color(color_gt)
            labels[i].set_fontweight('bold')
        else:

            labels[i].set_color('black')
            labels[i].set_fontweight('normal')
except Exception:
    pass


ax.set_yticks(y_pos)

ax.set_yticklabels(templates, fontsize=10)
ax.set_xlabel('Prompt Score', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

ax.set_xlim(0, max(scores) * 1.15)

output_path = 'failure_case_bigger_font.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")
plt.show()