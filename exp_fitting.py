"""Figure 1 experiment: how fast DIP fits an image, a noisy image, shuffled pixels and noise.

For every image and seed, DIP is fitted to 4 targets made from the same image,
and the full loss curve of each run is saved to results/fitting/.

Usage: .venv/bin/python exp_fitting.py            (full: 4 images x 4 targets x 3 seeds)
       .venv/bin/python exp_fitting.py --quick    (tiny test run for a CPU)
"""

import argparse
import os
import time

import dip
import utils

SIGMA = 25   # on the 0-255 scale, as utils.add_noise expects


def make_target(image, target, seed):
    """Build one of the 4 fitting targets from the clean image."""
    if target == "image":
        return image
    if target == "noisy":
        return utils.add_noise(image, sigma=SIGMA, seed=seed)
    if target == "shuffled":
        return utils.shuffle_pixels(image, seed=seed)
    if target == "noise":
        return utils.matched_noise(image, seed=seed)
    raise ValueError(f"Unknown target: {target}")


def first_iter_below(losses, threshold):
    """First iteration (1-based) where the loss is below threshold, or None."""
    for i in range(len(losses)):
        if losses[i] < threshold:
            return i + 1
    return None


def main():
    """Run every (image, target, seed) combination, skipping runs already saved."""
    parser = argparse.ArgumentParser(description="DIP fitting-speed experiment (Fig 1)")
    parser.add_argument("--images", nargs="+",
                        default=["parrots", "buildings", "river", "lighthouse"])
    parser.add_argument("--targets", nargs="+",
                        default=["image", "noisy", "shuffled", "noise"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--quick", action="store_true",
                        help="size 64, 100 iterations, parrots, seed 0 (for testing)")
    args = parser.parse_args()

    size = 256
    out_dir = "results/fitting"
    if args.quick:
        size = 64
        args.iters = 100
        args.images = ["parrots"]
        args.seeds = [0]
        out_dir = "results/fitting_quick"

    # List all runs first, so we know how many are left.
    runs = []
    for image_name in args.images:
        for target in args.targets:
            for seed in args.seeds:
                runs.append((image_name, target, seed))
    print(f"{len(runs)} runs, size {size}, {args.iters} iterations, saving to {out_dir}/")

    runs_done = 0       # runs actually computed in this session (for the time estimate)
    time_spent = 0.0
    for run_index, (image_name, target, seed) in enumerate(runs):
        name = f"{image_name}_{target}_s{seed}"
        path = os.path.join(out_dir, name + ".npz")
        runs_left = len(runs) - run_index - 1

        # Skip finished runs, so a restart after a Colab disconnect continues.
        if os.path.exists(path):
            print(f"skip {name} (already saved)")
            continue

        image = utils.load_image(f"images/{image_name}.png", size=size)
        target_image = make_target(image, target, seed)

        start_time = time.time()
        result = dip.run_dip(target_image, iters=args.iters, seed=seed, verbose=False)
        run_time = time.time() - start_time

        # Extra information for the report, stored in the same file.
        result["image"] = image_name
        result["target"] = target
        result["size"] = size
        # A value of None is not saved, so a missing key means "never reached".
        result["iter_below_0.01"] = first_iter_below(result["loss"], 0.01)
        result["iter_below_0.001"] = first_iter_below(result["loss"], 0.001)
        # Save to a temporary file first, then rename it. The rename is atomic,
        # so a disconnect while saving never leaves a half-written .npz behind.
        tmp_path = path + ".tmp.npz"
        dip.save_result(result, tmp_path)
        os.replace(tmp_path, path)

        runs_done += 1
        time_spent += run_time
        time_left = runs_left * time_spent / runs_done
        print(f"{name:28s} final loss {result['loss'][-1]:.5f}  "
              f"<0.01 at {result['iter_below_0.01']}  "
              f"<0.001 at {result['iter_below_0.001']}  "
              f"{run_time:6.1f} s  |  {runs_left} left, ~{time_left / 60:.1f} min")

    print("done")


if __name__ == "__main__":
    main()
