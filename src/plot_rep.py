import os
import re
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def find_epoch_files(directory: str, dataset_name: str) -> List[Tuple[int, str]]:
    """Return list of (epoch, filepath) sorted by epoch for files named epoch_{n}_rep_analysis.json."""
    files = []
    pat = re.compile(r"epoch_(\d+)_rep_analysis-"+dataset_name+r"\.json$")
    for name in os.listdir(directory):
        m = pat.match(name)
        if m:
            ep = int(m.group(1))
            files.append((ep, os.path.join(directory, name)))
    files.sort(key=lambda x: x[0])
    return files


essential_suffix = "_layer_mean_avg"

def load_category_matrix(directory: str, category: str, dataset_name: str) -> Tuple[np.ndarray, List[int]]:
    """Load a 2D matrix (epochs x layers) for the given category from epoch_* files.
    Returns (matrix, epochs) where matrix has shape (num_epochs, num_layers).
    """
    eps_files = find_epoch_files(directory, dataset_name)
    if not eps_files:
        raise FileNotFoundError(f"No epoch_*.json files found in {directory}")

    # First pass to determine minimal consistent layer count across epochs
    rows: List[List[float]] = []
    epoch_ids: List[int] = []
    min_layers = None
    key = f"{category}{essential_suffix}"

    for ep, fp in eps_files:
        with open(fp, "r") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        vals = summary.get(key)
        if vals is None:
            # category missing for this epoch; skip but keep alignment
            continue
        if min_layers is None:
            min_layers = len(vals)
        else:
            min_layers = min(min_layers, len(vals))

    if min_layers is None:
        raise ValueError(f"Category '{category}' not present in any epoch summaries under {directory}")

    # Second pass: collect rows, truncating to min_layers if needed
    for ep, fp in eps_files:
        with open(fp, "r") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        vals = summary.get(key)
        if vals is None:
            # fill with NaNs if missing
            row = [np.nan] * min_layers
        else:
            row = list(vals[:min_layers])
            if len(row) < min_layers:
                row = row + [np.nan] * (min_layers - len(row))
        rows.append(row)
        epoch_ids.append(ep)

    mat = np.array(rows, dtype=float)  # shape (E, L)
    return mat, epoch_ids


def plot_heatmap(ax, mat: np.ndarray, title: str, epochs: List[int], *, cmap: str = "viridis_r", vmin: float = 0.7, vmax: float = 1.0):
    # Draw heatmap with provided color scale
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Epoch")
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels(epochs)
    ax.set_xticks(range(mat.shape[1]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    parser = argparse.ArgumentParser(description="Plot representation similarity heatmaps across epochs and layers.")
    parser.add_argument("--dataset_name", type=str, default="Zebra_CoT", help="Name of the dataset.")
    parser.add_argument("--dir", help="Directory containing epoch_*_rep_analysis.json files", default="./sft_analysis")
    parser.add_argument("--out_dir", help="Output image path.", default="./sft_analysis/plot_imgs/")
    parser.add_argument("--cats", nargs="*", default=["observation_poss", "non_observation_poss"],
                        help="Categories to plot (must match *_layer_mean_avg keys in summaries)")
    args = parser.parse_args()

    cats = args.cats
    mats: Dict[str, np.ndarray] = {}
    epochs_ref: List[int] = []

    for c in cats:
        mat, epochs = load_category_matrix(args.dir, c, args.dataset_name)
        mats[c] = mat
        if not epochs_ref:
            epochs_ref = epochs
        else:
            # Align epochs if needed by truncation to common length
            common = min(len(epochs_ref), len(epochs))
            epochs_ref = epochs_ref[:common]
            mats[c] = mats[c][:common, :]
            # Also trim existing stored mats to common length
            for k in list(mats.keys()):
                if mats[k].shape[0] > common:
                    mats[k] = mats[k][:common, :]

    # Prepare optional difference: observation_poss - non_observation_poss
    add_diff = all(k in mats for k in ("observation_poss", "non_observation_poss"))
    n = len(cats) + (1 if add_diff else 0)

    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
    col = 0
    for c in cats:
        plot_heatmap(axes[0, col], mats[c], f"{c} (layer_mean_avg)", epochs_ref, cmap="viridis_r", vmin=0.7, vmax=1.0)
        col += 1

    if add_diff:
        diff = mats["non_observation_poss"] - mats["observation_poss"]
        # symmetric color limits around 0
        finite_vals = diff[np.isfinite(diff)]
        if finite_vals.size > 0:
            bound = float(np.nanmax(np.abs(finite_vals)))
            if bound == 0:
                bound = 1e-3
        else:
            bound = 1.0
        plot_heatmap(
            axes[0, col],
            diff,
            "non_observation_poss - observation_poss",
            epochs_ref,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )

    out_path = os.path.join(args.out_dir,f"rep_heatmaps{args.dataset_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

    # python ./src/plot_rep.py