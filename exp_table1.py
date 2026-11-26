"""Table 1 experiment: DIP against no prior, TV and BM3D on the test images.

For every sigma (15, 25, 50), test image and seed (0, 1, 2) this:
  1. makes the noisy image once and saves it, so every method sees the same noise,
  2. runs no prior, TV (tuned weight), TV (oracle weight) and BM3D on the CPU,
  3. runs DIP for 3000 iterations on the GPU and takes its output at the tuned
     stop ("DIP, fixed") and at its best iteration ("DIP, oracle").
TV weights, DIP stops and DIP settings are read from the tuning file of tune.py.
Every case is saved as soon as it is done, so a disconnect loses at most one.

Usage: python exp_table1.py [--sigmas S ...] [--images NAME ...] [--seeds K ...]
                            [--tuning FILE] [--quick]
Output in results/table1/: metrics/*.json (one per case), dip/*.npz, noisy/*.npy
"""

import argparse
import json
import os

import numpy as np
import torch

import baselines
import dip
import tune
import utils

TEST_IMAGES = ["buildings", "lighthouse", "parrots", "river"]
DEFAULT_TUNING = "results/tuning/tuning_jit0.033_ema0.99.json"


def get_noisy(clean, image_name, sigma, seed, out_dir):
    """Load the saved noisy image for this case, or make it and save it."""
    path = os.path.join(out_dir, "noisy", f"{image_name}_sigma{sigma}_seed{seed}.npy")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path))
    noisy = utils.add_noise(clean, sigma, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, noisy.numpy())   # .npy keeps the exact float values
    return noisy


def row(method, output, clean, runtime, device, param=None):
    """The results of one method on one case."""
    # float() turns numpy numbers into plain Python numbers, which json can save.
    return {"method": method,
            "psnr": float(utils.psnr(output, clean)),
            "ssim": float(utils.ssim(output, clean)),
            "runtime_s": float(runtime),
            "device": device,
            "param": param}   # TV weight or DIP stop iteration


def run_baselines(clean, noisy, sigma, tv_weight):
    """No prior, TV (tuned and oracle) and BM3D on one noisy image."""
    rows = []

    output, runtime = baselines.timed(baselines.no_prior, noisy)
    rows.append(row("No prior", output, clean, runtime, "cpu"))

    output, runtime = baselines.timed(baselines.tv_denoise, noisy, tv_weight)
    rows.append(row("TV (tuned)", output, clean, runtime, "cpu", tv_weight))

    # Oracle: the best weight for this very image, picked with the clean image.
    # It is an upper bound for TV, like "DIP (oracle)" is for DIP.
    best = None
    for weight in tune.TV_WEIGHTS:
        output, runtime = baselines.timed(baselines.tv_denoise, noisy, float(weight))
        score = utils.psnr(output, clean)
        if best is None or score > best["score"]:
            best = {"score": score, "weight": float(weight),
                    "output": output, "runtime": runtime}
    rows.append(row("TV (oracle)", best["output"], clean, best["runtime"], "cpu",
                    best["weight"]))

    output, runtime = baselines.timed(baselines.bm3d_denoise, noisy, sigma)
    rows.append(row("BM3D", output, clean, runtime, "cpu"))
    return rows


def run_dip_methods(clean, noisy, seed, stop, iters, settings, npz_path):
    """Run DIP once and return the rows for DIP (fixed) and DIP (oracle)."""
    if torch.cuda.is_available():
        device = "cuda: " + torch.cuda.get_device_name()
    else:
        device = "cpu"

    result = dip.run_dip(noisy, clean, iters=iters, seed=seed, save_at=(stop,),
                         save_path=npz_path, verbose=False, **settings)

    # We report the moving-average output if it is used, else the raw output.
    if settings["ema"] is not None:
        fixed_img = result["imgs_at_ema"][stop]
        best_img = result["best_ema_img"]
        best_iter = result["best_ema_iter"]
    else:
        fixed_img = result["imgs_at"][stop]
        best_img = result["best_img"]
        best_iter = result["best_iter"]

    # Runtime = time to reach that iteration (each iteration takes about as long).
    sec_per_iter = result["runtime_s"] / iters
    return [row("DIP (fixed)", fixed_img, clean, sec_per_iter * stop, device, stop),
            row("DIP (oracle)", best_img, clean, sec_per_iter * best_iter, device,
                best_iter)]


def main():
    """Run every (sigma, image, seed) case that is not saved yet."""
    parser = argparse.ArgumentParser(description="Table 1: DIP vs TV vs BM3D")
    parser.add_argument("--sigmas", nargs="+", type=int, default=[15, 25, 50])
    parser.add_argument("--images", nargs="+", default=TEST_IMAGES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--tuning", default=DEFAULT_TUNING)
    parser.add_argument("--out-dir", default="results/table1")
    parser.add_argument("--quick", action="store_true",
                        help="size 64, 60 iterations, one case (for testing)")
    args = parser.parse_args()

    size = 256
    if args.quick:
        size = 64
        args.iters = 60
        args.sigmas = [25]
        args.images = ["parrots"]
        args.seeds = [0]
        args.out_dir = "results/table1_quick"

    with open(args.tuning) as f:
        tuning = json.load(f)
    if not tuning["dip_stop"]:
        raise SystemExit(f"{args.tuning} has no DIP stops: run tune.py without --skip-dip")
    settings = tuning["dip_settings"]
    print("tuning file:", args.tuning)
    print("DIP settings:", settings)

    cases = [(sigma, name, seed) for sigma in args.sigmas
             for name in args.images for seed in args.seeds]
    print(f"{len(cases)} cases, size {size}, {args.iters} DIP iterations")

    for number, (sigma, name, seed) in enumerate(cases, start=1):
        case = f"{name}_sigma{sigma}_seed{seed}"
        metrics_path = os.path.join(args.out_dir, "metrics", case + ".json")
        if os.path.exists(metrics_path):
            print(f"[{number}/{len(cases)}] {case}: already done")
            continue

        clean = utils.load_image(f"images/{name}.png", size=size)
        noisy = get_noisy(clean, name, sigma, seed, args.out_dir)
        tv_weight = tuning["tv_weight"][str(sigma)]
        stop = min(tuning["dip_stop"][str(sigma)], args.iters)
        npz_path = os.path.join(args.out_dir, "dip", case + ".npz")

        rows = run_baselines(clean, noisy, sigma, tv_weight)
        rows += run_dip_methods(clean, noisy, seed, stop, args.iters, settings,
                                npz_path)
        for r in rows:
            r.update({"sigma": sigma, "image": name, "seed": seed})

        # Build the text first, then write it in one go, so a failed or
        # half-done case never leaves a file behind (it would count as done).
        text = json.dumps(rows, indent=2)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(text)

        summary = ", ".join(f"{r['method']} {r['psnr']:.2f}" for r in rows)
        print(f"[{number}/{len(cases)}] {case}: {summary}")


if __name__ == "__main__":
    main()
