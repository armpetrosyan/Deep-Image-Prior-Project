"""Make Figure 1 (fitting speed) and its table from the exp_fitting.py results.

Usage: .venv/bin/python plot_fig1.py            (reads results/fitting/)
       .venv/bin/python plot_fig1.py --quick    (reads results/fitting_quick/)
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")   # draw to files only, no window needed (works on Colab)
import matplotlib.pyplot as plt
import numpy as np

import dip

TARGETS = ["image", "noisy", "shuffled", "noise"]
LABELS = {"image": "Natural image", "noisy": "Image + noise (σ=25)",
          "shuffled": "Shuffled pixels", "noise": "Gaussian noise"}
# Colour plus a line style, so the lines are still distinct in grayscale.
COLORS = {"image": "#257de0", "noisy": "#f47230",
          "shuffled": "#00b894", "noise": "#f2a600"}
STYLES = {"image": "-", "noisy": "--", "shuffled": "-.", "noise": ":"}
THRESHOLDS = ["0.01", "0.001"]
NOISE_VARIANCE = (25 / 255) ** 2   # MSE between a clean and a sigma=25 noisy image

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_runs(folder):
    """Load every finished .npz in folder (leftover .tmp.npz files are skipped)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        if path.endswith(".tmp.npz"):
            continue
        result = dip.load_result(path)
        run = {
            "image": str(result["image"]),
            "target": str(result["target"]),
            "seed": int(result["seed"]),
            "loss": result["loss"],
        }
        # A missing key means the loss never went below that threshold.
        for t in THRESHOLDS:
            key = f"iter_below_{t}"
            if key in result:
                run[key] = int(result[key])
            else:
                run[key] = None
        runs.append(run)
    return runs


def plot_curves(ax, runs, target):
    """Plot the mean loss curve of the runs for one target, with a min-max band."""
    curves = []
    for run in runs:
        if run["target"] == target:
            curves.append(run["loss"])
    if len(curves) == 0:
        return
    curves = np.stack(curves)              # shape [num_runs, num_iters]
    iterations = np.arange(1, curves.shape[1] + 1)
    color = COLORS[target]
    ax.fill_between(iterations, curves.min(axis=0), curves.max(axis=0),
                    color=color, alpha=0.2, linewidth=0)
    ax.plot(iterations, curves.mean(axis=0), color=color,
            linestyle=STYLES[target], linewidth=2, label=LABELS[target])


def style_axes(ax):
    """Log x axis like the paper's learning curves, light grid."""
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    ax.set_ylim(bottom=0)
    # Show iteration numbers on every panel, even when panels share the x axis.
    ax.tick_params(labelbottom=True)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)


def draw_noise_level(ax):
    """Thin grey dashed line at the noise variance, for reference."""
    ax.axhline(NOISE_VARIANCE, color="#888888", linestyle="--", linewidth=0.8,
               label="noise variance (σ=25)")


def save_figure(fig, out_dir, name):
    """Save a figure as PNG (dpi 200) and PDF."""
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main_figure(runs, out_dir):
    """All images and seeds together: one line per target."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for target in TARGETS:
        plot_curves(ax, runs, target)
    draw_noise_level(ax)
    style_axes(ax)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "fig1_fitting")


def per_image_figure(runs, out_dir):
    """One panel per image (2x2 grid), averaged over seeds only."""
    image_names = sorted(set(run["image"] for run in runs))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = axes.flatten()
    for i in range(4):
        ax = axes[i]
        if i >= len(image_names):
            ax.axis("off")         # fewer than 4 images: leave the panel empty
            continue
        image_runs = []
        for run in runs:
            if run["image"] == image_names[i]:
                image_runs.append(run)
        for target in TARGETS:
            plot_curves(ax, image_runs, target)
        draw_noise_level(ax)
        style_axes(ax)
        if i < 2:
            ax.set_xlabel("")      # "Iteration" only on the bottom row
        ax.set_title(image_names[i])
    axes[0].legend(frameon=False, fontsize=9)
    save_figure(fig, out_dir, "fig1_per_image")


def make_table(runs):
    """Mean and std of the first iteration below each threshold, per target."""
    rows = []
    for target in TARGETS:
        target_runs = []
        for run in runs:
            if run["target"] == target:
                target_runs.append(run)
        if len(target_runs) == 0:
            continue
        row = {"target": target}
        for t in THRESHOLDS:
            reached = []
            for run in target_runs:
                if run[f"iter_below_{t}"] is not None:
                    reached.append(run[f"iter_below_{t}"])
            never = len(target_runs) - len(reached)
            # Mean and std only over the runs that did reach the threshold.
            if len(reached) > 0:
                row[f"mean_{t}"] = f"{np.mean(reached):.0f}"
                row[f"std_{t}"] = f"{np.std(reached):.0f}"
            else:
                row[f"mean_{t}"] = "-"
                row[f"std_{t}"] = "-"
            row[f"never_{t}"] = f"{never}/{len(target_runs)}"
        rows.append(row)
    return rows


def print_and_save_table(rows, out_dir):
    """Print the table in the terminal and save it as a CSV file."""
    columns = ["target"]
    for t in THRESHOLDS:
        columns += [f"mean_{t}", f"std_{t}", f"never_{t}"]

    print("First iteration with loss below the threshold (mean ± std over runs that reached it)")
    print(f"{'target':10s} | {'< 0.01':>12s} {'never':>7s} | {'< 0.001':>12s} {'never':>7s}")
    for row in rows:
        line = f"{row['target']:10s} |"
        for t in THRESHOLDS:
            value = f"{row[f'mean_{t}']} ± {row[f'std_{t}']}"
            line += f" {value:>12s} {row[f'never_{t}']:>7s} |"
        print(line.rstrip(" |"))

    with open(os.path.join(out_dir, "fig1_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Load the results, then make both figures and the table."""
    parser = argparse.ArgumentParser(description="Make Fig 1 from exp_fitting.py results")
    parser.add_argument("--dir", default="results/fitting")
    parser.add_argument("--quick", action="store_true",
                        help="read results/fitting_quick instead")
    args = parser.parse_args()

    folder = args.dir
    out_dir = "results/fig1"
    if args.quick:
        folder = "results/fitting_quick"
        out_dir = "results/fig1_quick"   # a test never overwrites the real figures

    runs = load_runs(folder)
    if len(runs) == 0:
        print(f"No results found in {folder}/")
        return
    print(f"Loaded {len(runs)} runs from {folder}/")

    os.makedirs(out_dir, exist_ok=True)
    main_figure(runs, out_dir)
    per_image_figure(runs, out_dir)
    print_and_save_table(make_table(runs), out_dir)
    print(f"Saved figures and table to {out_dir}/")


if __name__ == "__main__":
    main()
