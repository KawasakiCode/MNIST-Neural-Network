"""Shared matplotlib style + colors so every plot in the project looks consistent."""
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt

# One color per framework, reused across every plot.
COLORS = {
    "numpy": "#4C72B0",
    "pytorch": "#DD8452",
    "tensorflow": "#55A868",
}

LABELS = {
    "numpy": "NumPy (from scratch)",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })
