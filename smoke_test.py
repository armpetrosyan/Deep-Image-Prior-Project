"""End-to-end test of the DIP pipeline, plus a timer to plan the full runs.

Usage: .venv/bin/python smoke_test.py [--image PATH] [--size N] [--iters N]
                                      [--sigma S] [--seed K]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw to files only, no window needed (works on Colab)
import matplotlib.pyplot as plt
import torch

import dip
import utils


def save_comparison(clean_image, noisy_image, noisy_psnr, result, title, path):
    """Save a figure with the clean, noisy and best DIP images side by side."""
    best_title = f"DIP best, iter {result['best_iter']} ({result['best_psnr']:.1f} dB)"
    images = [clean_image, noisy_image, result["best_img"]]
    titles = ["Clean", f"Noisy ({noisy_psnr:.1f} dB)", best_title]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for i in range(3):
        axes[i].imshow(utils.to_numpy(images[i].clamp(0, 1)))
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.suptitle(title)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run DIP once, report timing and PSNR, save a figure and check the results."""
    parser = argparse.ArgumentParser(description="DIP smoke test")
    parser.add_argument("--image", default="images/parrots.png")
    # Default None so we can tell whether the user set these themselves.
    parser.add_argument("--size", type=int, default=None, help="default 256")
    parser.add_argument("--iters", type=int, default=None, help="default 200")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print("device: cuda")
        print("GPU:", torch.cuda.get_device_name())
    else:
        print("device: cpu")

    size = args.size
    num_iters = args.iters
    if not has_gpu and size is None and num_iters is None:
        size = 64
        num_iters = 100
        print("note: no GPU, so using size 64 and 100 iterations. A CPU is too "
              "slow for the real size; real runs belong on Colab.")
    if size is None:
        size = 256
    if num_iters is None:
        num_iters = 200
    print(f"image={args.image} size={size} iters={num_iters} "
          f"sigma={args.sigma} seed={args.seed}")

    clean_image = utils.load_image(args.image, size=size)
    noisy_image = utils.add_noise(clean_image, sigma=args.sigma, seed=args.seed)
    noisy_psnr = utils.psnr(noisy_image, clean_image)
    print(f"noisy input PSNR: {noisy_psnr:.2f} dB")

    npz_path = "results/smoke_test.npz"
    result = dip.run_dip(noisy_image, clean_image, iters=num_iters, seed=args.seed,
                         log_every=10, save_at=(num_iters,), save_path=npz_path)

    first_loss = result["loss"][0]
    last_loss = result["loss"][-1]
    print(f"first loss {first_loss:.5f}, last loss {last_loss:.5f}")
    print(f"best PSNR {result['best_psnr']:.2f} dB at iter {result['best_iter']}")

    # Time estimates for planning: one full run, and Person A's
    # 48 runs (4 targets x 4 images x 3 seeds).
    sec_per_iter = result["runtime_s"] / num_iters
    one_run_sec = sec_per_iter * 3000
    all_runs_sec = 48 * one_run_sec
    print(f"{sec_per_iter:.3f} s/iter at size {size}")
    print(f"estimate: one 3000-iter run {one_run_sec / 60:.1f} min, "
          f"48 runs {all_runs_sec / 3600:.1f} h")

    fig_path = "results/smoke_test/comparison.png"
    fig_title = (f"{os.path.basename(args.image)}, sigma={args.sigma:g}, "
                 f"{num_iters} iterations")
    save_comparison(clean_image, noisy_image, noisy_psnr, result, fig_title, fig_path)

    checks = {
        "loss went down": last_loss < first_loss,
        "best PSNR > noisy PSNR": result["best_psnr"] > noisy_psnr,
        "output files exist": os.path.exists(npz_path) and os.path.exists(fig_path),
    }
    all_passed = True
    for name, passed in checks.items():
        if passed:
            print(f"PASS: {name}")
        else:
            print(f"FAIL: {name}")
            all_passed = False
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
