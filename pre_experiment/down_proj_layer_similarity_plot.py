"""
down_proj_layer_similarity_plot.py

Plots the mean cosine similarity of down_proj input activations
across all 32 layers of Llama-3-8B-Instruct as a scatter plot.

Usage:
    python pre_experiment/down_proj_layer_similarity_plot.py
"""

import matplotlib
matplotlib.use("Agg")  # headless / server-safe backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (layer, mean_cosine_similarity)
# ---------------------------------------------------------------------------
DATA = [
    (0,  0.9919774532318115),
    (1,  0.9896363615989685),
    (2,  0.9858349561691284),
    (3,  0.9615794420242310),
    (4,  0.9447027444839478),
    (5,  0.9323343634605408),
    (6,  0.9137963652610779),
    (7,  0.8946613073348999),
    (8,  0.8833200335502625),
    (9,  0.8710764050483704),
    (10, 0.8659445047378540),
    (11, 0.8569672107696533),
    (12, 0.8716274499893188),
    (13, 0.8855525255203247),
    (14, 0.8955749273300171),
    (15, 0.8829154968261719),
    (16, 0.8826772570610046),
    (17, 0.8601453304290771),
    (18, 0.8337495326995850),
    (19, 0.8324052691459656),
    (20, 0.8078095316886902),
    (21, 0.8126136660575867),
    (22, 0.8016081452369690),
    (23, 0.7765399813652039),
    (24, 0.7798742651939392),
    (25, 0.7974036931991577),
    (26, 0.7786945104598999),
    (27, 0.8076067566871643),
    (28, 0.8278945684432983),
    (29, 0.8463263511657715),
    (30, 0.8522738814353943),
    (31, 0.8886415958404541),
]

layers = np.array([d[0] for d in DATA])
sims   = np.array([d[1] for d in DATA])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Scatter points — colour-coded by similarity (viridis)
sc = ax.scatter(layers, sims, c=sims, cmap="viridis", s=80, zorder=5,
                edgecolors="white", linewidths=0.6)

# Light connecting line to show trend
ax.plot(layers, sims, color="steelblue", linewidth=1.2,
        alpha=0.45, zorder=4)

# Highlight the minimum
min_idx = int(np.argmin(sims))
ax.scatter(layers[min_idx], sims[min_idx], s=140, zorder=6,
           edgecolors="crimson", linewidths=1.8, facecolors="none",
           label=f"Min  (layer {layers[min_idx]:d}, {sims[min_idx]:.4f})")

# Highlight the maximum
max_idx = int(np.argmax(sims))
ax.scatter(layers[max_idx], sims[max_idx], s=140, zorder=6,
           edgecolors="darkorange", linewidths=1.8, facecolors="none",
           label=f"Max  (layer {layers[max_idx]:d}, {sims[max_idx]:.4f})")

# Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Mean Cosine Similarity", fontsize=11)

# Labels and formatting
ax.set_xlabel("Layer Index", fontsize=13)
ax.set_ylabel("Mean Cosine Similarity", fontsize=13)
ax.set_title(
    "Prompt-Activation Alignment of Llama-3-8B-Instruct",
    fontsize=14, fontweight="bold"
)
ax.set_xticks(layers)
ax.xaxis.set_tick_params(labelsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.set_xlim(-0.8, 31.8)
ax.set_ylim(0.74, 1.01)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(fontsize=10, framealpha=0.85)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path("pre_experiment")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "down_proj_layer_similarity_scatter.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)
