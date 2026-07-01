"""
Loss & accuracy training curves for every framework.

Reads the per-epoch history JSON files written by the three train.py scripts:
    metrics/history_numpy.json
    metrics/history_pytorch.json
    metrics/history_tensorflow.json

Missing files are skipped (with a warning), so you can plot whatever you have.

Outputs:
    plots/img/loss_curves.png       - loss vs epoch, all frameworks
    plots/img/accuracy_curves.png   - train accuracy vs epoch, all frameworks

Run:  python plot_curves.py   (from inside the plots/ directory)
"""
import os
import json
from _style import apply_style, COLORS, LABELS
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
METRICS = os.path.join(HERE, "..", "metrics")
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)
apply_style()

FRAMEWORKS = ["numpy", "pytorch", "tensorflow"]


def load_histories():
    histories = {}
    for fw in FRAMEWORKS:
        path = os.path.join(METRICS, f"history_{fw}.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[fw] = json.load(f)
        else:
            print(f"[skip] {path} not found - run {fw}/train.py to generate it")
    return histories


def plot_metric(histories, key, ylabel, title, filename, ylim=None):
    if not histories:
        print("No history files found; nothing to plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fw, hist in histories.items():
        ax.plot(hist["epoch"], hist[key], "-o", markersize=3,
                color=COLORS[fw], linewidth=2, label=LABELS[fw])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    path = os.path.join(OUT, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    histories = load_histories()
    plot_metric(histories, "loss", "Training loss (cross-entropy)",
                "Training Loss per Epoch", "loss_curves.png")
    plot_metric(histories, "accuracy", "Training accuracy (%)",
                "Training Accuracy per Epoch", "accuracy_curves.png")
