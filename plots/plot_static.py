"""
Static README plots that do NOT need any training run.

They are generated straight from the results already documented in README.md:
  1. architecture_evolution.png - the NumPy step-by-step accuracy story (steps 1-8)
  2. framework_comparison.png   - final train/test accuracy for NumPy / PyTorch / TensorFlow

Run:  python plot_static.py   (from inside the plots/ directory)
"""
import os
import numpy as np
from _style import apply_style, COLORS, LABELS
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(OUT, exist_ok=True)
apply_style()

# --------------------------------------------------------------------------- #
# 1. NumPy architecture evolution (numbers taken from README.md, steps 1-8)
#    Early steps only report an approximate test accuracy, so train is None.
# --------------------------------------------------------------------------- #
STEPS = [
    ("1. Shallow\n(128)",        None,  89.0),
    ("2. Wide\n(512)",           None,  89.0),
    ("3. Deep\n(128->128)",      None,  91.0),
    ("4. CNN",                   97.20, 96.67),
    ("5. Adam\n+ LR decay",      99.95, 98.09),
    ("6. Data\naugmentation",    91.94, 98.59),
    ("7. Dropout",               97.15, 98.67),
    ("8. Max\npooling",          98.76, 99.31),
]


def plot_architecture_evolution():
    names = [s[0] for s in STEPS]
    train = [s[1] for s in STEPS]
    test = [s[2] for s in STEPS]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Test accuracy is the through-line of the whole story -> emphasize it.
    ax.plot(x, test, "-o", color=COLORS["numpy"], linewidth=2.5,
            markersize=7, label="Test accuracy", zorder=3)
    # Train accuracy where it was recorded.
    tx = [i for i, t in enumerate(train) if t is not None]
    ty = [train[i] for i in tx]
    ax.plot(tx, ty, "--s", color="#C44E52", linewidth=1.8,
            markersize=6, label="Train accuracy", alpha=0.9, zorder=2)

    for xi, t in zip(x, test):
        ax.annotate(f"{t:.1f}", (xi, t), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9,
                    color=COLORS["numpy"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 101)
    ax.set_title("NumPy From-Scratch Network: Accuracy Across Architecture Evolution",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(99, color="gray", linestyle=":", alpha=0.5)
    ax.text(0, 99.1, "99% barrier", fontsize=8, color="gray")

    path = os.path.join(OUT, "architecture_evolution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# 2. Final train/test accuracy per framework (from README.md)
# --------------------------------------------------------------------------- #
FINAL = {
    "numpy":      (98.76, 99.31),
    "pytorch":    (98.38, 99.06),
    "tensorflow": (99.71, 99.06),
}


def plot_framework_comparison():
    frameworks = list(FINAL.keys())
    train = [FINAL[f][0] for f in frameworks]
    test = [FINAL[f][1] for f in frameworks]
    x = np.arange(len(frameworks))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, train, w, label="Train accuracy",
                color=[COLORS[f] for f in frameworks], alpha=0.55,
                edgecolor="white")
    b2 = ax.bar(x + w / 2, test, w, label="Test accuracy",
                color=[COLORS[f] for f in frameworks], edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in frameworks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(95, 100.5)
    ax.set_title("Final Accuracy by Framework (identical architecture)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    # Faint bar fill = train, solid = test; note it for readers.
    ax.text(0.01, 0.02, "faint bar = train   |   solid bar = test",
            transform=ax.transAxes, fontsize=8, color="gray")

    path = os.path.join(OUT, "framework_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    plot_architecture_evolution()
    plot_framework_comparison()
