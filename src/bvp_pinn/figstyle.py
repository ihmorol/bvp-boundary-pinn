"""One figure style for the reference figures and the PINN result figures.

Rougier, Droettboom and Bourne (2014), "Ten simple rules for better figures"
(doi:10.1371/journal.pcbi.1003833) and Wilke, Fundamentals of Data Visualization, ch. 20: one
message per figure, the Okabe-Ito colour-blind-safe palette, a second cue (line style or marker)
wherever colour separates series.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "orange": "#E69F00", "sky": "#56B4E9", "black": "#000000", "grey": "#7F7F7F"}
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
         "pdf.fonttype": 42, "figure.dpi": 100}
MM = 1 / 25.4
WIDE = 180 * MM


def save(fig, path):
    """PNG at 300 dpi and a vector PDF next to it."""
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def letters(axes):
    """Panel letters (a), (b), ... at the top left of each panel."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(-0.02, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold")
