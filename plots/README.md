# Plots

All figures used in the project README. Every script writes PNGs into `plots/img/`.

> Run the scripts **from inside this `plots/` directory**. The repo root contains a
> local `numpy/` package that shadows the real NumPy install; running from a
> subdirectory keeps the real NumPy on the import path.

```bash
cd plots
python plot_static.py        # architecture evolution, framework comparison, generalization gap (no training needed)
python plot_curves.py        # loss / accuracy curves (needs metrics/*.json)
python plot_diagnostics.py   # confusion matrix, per-class acc, misclassifications, filters, activation maps
```

## What needs what

| Script | Inputs | Needs a training run? |
|---|---|---|
| `plot_static.py` | numbers hard-coded from README results | No |
| `plot_diagnostics.py` | saved weights (`trained_model.pth`) + test CSV | No — inference only |
| `plot_curves.py` | `metrics/history_{numpy,pytorch,tensorflow}.json` | Yes — see below |

## Generating the loss/accuracy curves

The three `train.py` scripts now append per-epoch metrics to
`metrics/history_<framework>.json`. Those files do not exist until you train, so
run each framework's training once (in the environment where it works), then plot:

```bash
cd numpy      && python train.py     # writes metrics/history_numpy.json
cd pytorch    && python train.py     # writes metrics/history_pytorch.json
cd tensorflow && python train.py     # writes metrics/history_tensorflow.json
cd plots      && python plot_curves.py
```

`plot_curves.py` skips any framework whose history file is missing, so you can
plot whatever you have.

## Extending diagnostics to NumPy / TensorFlow

`plot_diagnostics.py` is currently wired to the PyTorch model because that is the
framework installed here. To add the other two, load their saved weights
(`numpy/trained_weights.npz`, `trained_model.weights.h5`), run their forward pass
to get `y_pred` and the conv filters, and reuse the same plotting functions.
