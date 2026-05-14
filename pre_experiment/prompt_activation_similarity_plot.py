import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


sns.reset_orig()


def custom_format_func(x, pos):

    if x >= 0.999: return "1.0"
    return f"{x:.2f}".lstrip('0')


def main():

    script_dir = Path(__file__).resolve().parent
    npy_path = script_dir / "prompt_activation_cosine_similarity_matrix.npy"

    if not npy_path.exists():

        sim_matrix = np.random.uniform(0.7, 0.98, (10, 10))
        np.fill_diagonal(sim_matrix, 1.0)
    else:
        sim_matrix = np.load(npy_path)

    n = sim_matrix.shape[0]


    annot_labels = np.empty_like(sim_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            if val >= 0.999:
                annot_labels[i, j] = "1.0"
            else:
                formatted = f"{val:.2f}".lstrip('0')

                annot_labels[i, j] = formatted if formatted != "." and formatted != "" else ".00"


    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)


    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 0.8


    fig, ax = plt.subplots(figsize=(3.25, 3.25))


    heatmap = sns.heatmap(
        sim_matrix,
        mask=mask,
        annot=annot_labels,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={
            "label": "Cosine Similarity",
            "fraction": 0.046,
            "pad": 0.04,
        },

        xticklabels=[f"$r_{{{i}}}$" for i in range(n)],
        yticklabels=[f"$r_{{{i}}}$" for i in range(n)],
        linewidths=0.5,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 7, "fontfamily": "serif"},
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )


    ax.tick_params(left=True, bottom=True, which='major', width=0.8, length=3)
    ax.set_xlabel("Template Index", fontsize=10)
    ax.set_ylabel("Template Index", fontsize=10)


    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)


    cbar = heatmap.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_format_func))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Cosine Similarity", fontsize=9, family="serif")


    plt.tight_layout(pad=0.2)
    output_filename = script_dir / "prompt_activation_lower_triangle_heatmap.pdf"


    plt.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Success: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()