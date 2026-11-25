"""Tune the TV weight and the DIP fixed stop on the tuning image only.

For each sigma (15, 25, 50) this picks:
  - the TV weight with the best mean PSNR over seeds (grid search, CPU, seconds),
  - the DIP stop iteration with the best mean PSNR over seeds (GPU, ~minutes).
The test images are never used here, so "TV, tuned" and "DIP, fixed" are fair.

Usage: python tune.py [--skip-dip] [--iters N] [--size N] [--out-dir DIR]
                      [--jitter J] [--ema E]
Defaults are the paper's settings (jitter 1/30, ema 0.99); 0 turns either off.
Output: <out-dir>/tuning_<settings>.json, <out-dir>/tuning_<settings>.png and
one .npz per DIP run, where <settings> names the DIP settings (see settings_tag).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # draw to files only, no window needed (works on Colab)
import matplotlib.pyplot as plt
import numpy as np

import baselines
import dip
import utils

TUNING_IMAGE = "images/tuning_plane.png"
SIGMAS = [15, 25, 50]
SEEDS = [0, 1, 2]
TV_WEIGHTS = np.round(np.arange(0.02, 0.305, 0.01), 2)   # 0.02, 0.03, ..., 0.30

# DIP settings. They must be the same as in the Table 1 experiment.
# input_noise (jitter on z) and ema (moving average of the outputs) are set
# from the command line in main(); the paper uses 1/30 and 0.99.
DIP_SETTINGS = {"lr": 0.01, "log_every": 10, "input_noise": 1 / 30, "ema": 0.99}


def settings_tag():
    """Short text naming the DIP settings, used in file names, e.g. 'jit0.033_ema0.99'."""
    return f"jit{DIP_SETTINGS['input_noise']:.3f}_ema{DIP_SETTINGS['ema']}"


def psnr_curve(result):
    """The PSNR curve we report: of the moving average if it is used, else of the raw output."""
    if DIP_SETTINGS["ema"] is not None:
        return result["psnr_ema"]
    return result["psnr"]


def tune_tv(clean, noisy_images):
    """Return (best weight, mean PSNR for every weight) for one sigma."""
    mean_psnrs = []
    for weight in TV_WEIGHTS:
        scores = [utils.psnr(baselines.tv_denoise(x0, weight), clean)
                  for x0 in noisy_images]
        mean_psnrs.append(np.mean(scores))
    mean_psnrs = np.array(mean_psnrs)
    best_weight = float(TV_WEIGHTS[np.argmax(mean_psnrs)])
    return best_weight, mean_psnrs


def dip_curve(clean, x0, sigma, seed, iters, out_dir):
    """Run DIP once (or load the saved run) and return (psnr_iters, psnrs)."""
    # The settings are in the file name, so runs with other settings are never reused.
    name = f"dip_{settings_tag()}_sigma{sigma}_seed{seed}.npz"
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        saved = dip.load_result(path)
        if int(saved["iters"]) == iters:   # reuse only a run of the same length
            print(f"  sigma={sigma} seed={seed}: loaded {path}")
            return saved["psnr_iters"], psnr_curve(saved)

    print(f"  sigma={sigma} seed={seed}: running DIP for {iters} iterations")
    result = dip.run_dip(x0, clean, iters=iters, seed=seed, save_path=path,
                         **DIP_SETTINGS)
    return result["psnr_iters"], psnr_curve(result)


def tune_dip(clean, noisy_images, sigma, iters, out_dir):
    """Return (best stop, logged iterations, mean PSNR curve) for one sigma."""
    curves = []
    for seed, x0 in zip(SEEDS, noisy_images):
        psnr_iters, psnrs = dip_curve(clean, x0, sigma, seed, iters, out_dir)
        curves.append(psnrs)
    mean_curve = np.mean(curves, axis=0)
    best_stop = int(psnr_iters[np.argmax(mean_curve)])
    return best_stop, psnr_iters, mean_curve


def main():
    """Tune both methods for every sigma, then save the choices and a plot."""
    parser = argparse.ArgumentParser(description="Tune TV weight and DIP stop")
    parser.add_argument("--skip-dip", action="store_true", help="only tune TV")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/tuning")
    parser.add_argument("--jitter", type=float, default=1 / 30,
                        help="std of the noise added to z each step (0 = off)")
    parser.add_argument("--ema", type=float, default=0.99,
                        help="moving-average weight of the outputs (0 = off)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Settings from the command line, so other settings need no code change.
    DIP_SETTINGS["input_noise"] = args.jitter
    if args.ema > 0:
        DIP_SETTINGS["ema"] = args.ema
    else:
        DIP_SETTINGS["ema"] = None

    clean = utils.load_image(TUNING_IMAGE, size=args.size)
    choices = {"image": TUNING_IMAGE, "dip_settings": DIP_SETTINGS,
               "tv_weight": {}, "dip_stop": {}}
    tv_curves = {}
    dip_curves = {}

    for sigma in SIGMAS:
        # Same noise convention as everywhere else: the seed sets the noise.
        noisy_images = [utils.add_noise(clean, sigma, seed=s) for s in SEEDS]

        weight, tv_curves[sigma] = tune_tv(clean, noisy_images)
        choices["tv_weight"][str(sigma)] = weight
        print(f"sigma={sigma}: best TV weight {weight:.2f} "
              f"({tv_curves[sigma].max():.2f} dB)")

        if not args.skip_dip:
            stop, psnr_iters, curve = tune_dip(clean, noisy_images, sigma,
                                               args.iters, args.out_dir)
            choices["dip_stop"][str(sigma)] = stop
            dip_curves[sigma] = (psnr_iters, curve)
            print(f"sigma={sigma}: best DIP stop {stop} ({curve.max():.2f} dB)")

    # Name the outputs after the settings too, so old results are kept for comparison.
    json_path = os.path.join(args.out_dir, f"tuning_{settings_tag()}.json")
    with open(json_path, "w") as f:
        json.dump(choices, f, indent=2)
    print("saved", json_path)
    print(json.dumps(choices, indent=2))

    # Plot: PSNR against TV weight, and PSNR against DIP iteration.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sigma in SIGMAS:
        axes[0].plot(TV_WEIGHTS, tv_curves[sigma], label=f"sigma={sigma}")
        if sigma in dip_curves:
            psnr_iters, curve = dip_curves[sigma]
            axes[1].plot(psnr_iters, curve, label=f"sigma={sigma}")
            axes[1].axvline(choices["dip_stop"][str(sigma)], linestyle="--",
                            color=axes[1].lines[-1].get_color())
    axes[0].set_xlabel("TV weight")
    axes[0].set_title("TV on the tuning image")
    axes[1].set_xlabel("DIP iteration")
    axes[1].set_title("DIP on the tuning image (dashed = chosen stop)")
    for ax in axes:
        ax.set_ylabel("mean PSNR over seeds (dB)")
        if ax.lines:   # the DIP panel is empty with --skip-dip
            ax.legend()
    plot_path = os.path.join(args.out_dir, f"tuning_{settings_tag()}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", plot_path)


if __name__ == "__main__":
    main()
