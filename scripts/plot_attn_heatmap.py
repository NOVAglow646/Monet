#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

"""
Visualize per-layer attention stats over steps as heatmaps.
- Input files: ./logs/attn_analysis/attn_layers_step_XXXXXXXX.npz
- Each npz contains arrays: mean_qimg, sum_qimg, mean_img_rest, sum_img_rest,
  mean_latent, sum_latent, mean_other_prev, sum_other_prev, and 'step'.
- We render one figure with 8 subplots (2 rows x 4 cols):
  for each flow (qimg, img_rest, latent, other_prev), left=mean, right=sum.
- X axis: step index (sorted by filename step); Y axis: layer index.
"""

SAVE_DIR = os.environ.get('ATTN_PLOT_DIR', './logs/attn_analysis/figs')
SRC_DIR = os.environ.get('ATTN_SRC_DIR', './logs/attn_analysis')

def load_series(src_dir: str):
    paths = sorted(glob.glob(os.path.join(src_dir, 'attn_layers_step_*.npz')))
    steps = []
    series = {}
    for p in paths:
        try:
            data = np.load(p)
            step = int(data.get('step', 0))
            steps.append(step)
            for k in ['mean_qimg','sum_qimg','mean_img_rest','sum_img_rest','mean_latent','sum_latent','mean_other_prev','sum_other_prev']:
                arr = data.get(k, None)
                if arr is None:
                    continue
                series.setdefault(k, []).append(arr.astype(float))
        except Exception:
            continue
    return steps, series


def to_matrix(arr_list):
    # arr_list: list of 1D arrays (layers,) per step -> stack to (layers, steps)
    if not arr_list:
        return None
    try:
        mat = np.stack(arr_list, axis=1)  # (L, T)
        return mat
    except Exception:
        # Align by min length as fallback
        L = min(len(a) for a in arr_list)
        arr_list = [a[:L] for a in arr_list]
        return np.stack(arr_list, axis=1)


def plot_heatmaps(steps, series, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Build matrices
    keys = [
        ('mean_qimg','Question image (mean)'), ('sum_qimg','Question image (sum)'),
        ('mean_img_rest','Rest images (mean)'), ('sum_img_rest','Rest images (sum)'),
        ('mean_latent','Latent (mean)'), ('sum_latent','Latent (sum)'),
        ('mean_other_prev','Other-prev (mean)'), ('sum_other_prev','Other-prev (sum)')
    ]
    mats = []
    for k, _ in keys:
        mats.append(to_matrix(series.get(k, [])))

    # Determine common layer count for display
    L = 0
    for m in mats:
        if m is not None:
            L = max(L, m.shape[0])
    T = len(steps)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for idx, (k, title) in enumerate(keys):
        r, c = divmod(idx, 4)
        ax = axes[r, c]
        mat = to_matrix(series.get(k, []))
        if mat is None:
            ax.set_title(title + ' (no data)')
            ax.axis('off')
            continue
        im = ax.imshow(mat, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('step (index)')
        ax.set_ylabel('layer')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = os.path.join(save_dir, 'attn_over_steps.png')
    fig.suptitle('Per-layer attention over steps')
    fig.savefig(out_path, dpi=200)
    print(f'Saved: {out_path}')


def main():
    steps, series = load_series(SRC_DIR)
    if not steps:
        print('No attention NPZ files found in', SRC_DIR)
        return
    plot_heatmaps(steps, series, SAVE_DIR)

if __name__ == '__main__':
    main()
