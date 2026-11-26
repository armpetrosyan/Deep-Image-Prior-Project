"""Build Table 1 from the per-case results of exp_table1.py.

Table 1: one row per method, and for every sigma the PSNR, SSIM (mean +- std
over all test images and seeds) and the mean runtime.
Appendix table: the same per image (mean +- std over seeds).

Usage: python make_table1.py [--in-dir results/table1]
Output in <in-dir>/: table1.csv, table1.tex, per_image.csv, per_image.tex
The .tex files need \\usepackage{booktabs} (the NeurIPS template has it).
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

METHODS = ["No prior", "TV (tuned)", "TV (oracle)", "BM3D", "DIP (fixed)", "DIP (oracle)"]
SIGMAS = [15, 25, 50]


def load_rows(folder):
    """All result rows from every case file in folder."""
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as f:
            rows += json.load(f)
    return rows


def select(rows, **conditions):
    """The rows that match every condition, e.g. select(rows, method="BM3D", sigma=25)."""
    return [r for r in rows if all(r[key] == value for key, value in conditions.items())]


def stats(rows, metric):
    """Mean and standard deviation of one metric over some rows."""
    values = np.array([r[metric] for r in rows])
    # ddof=1: the sample standard deviation (the spread between runs).
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return values.mean(), std


def fmt_time(seconds):
    """Runtime as short text: '<0.01', '0.05', '13.1', '107'."""
    if seconds < 0.01:
        return "<0.01"
    return f"{seconds:.3g}"


def summary(rows, images):
    """One dict per (method, sigma): mean and std of PSNR, SSIM and runtime."""
    table = []
    for method in METHODS:
        for sigma in SIGMAS:
            cell = select(rows, method=method, sigma=sigma)
            if len(cell) != len(images) * 3:
                print(f"WARNING: {method}, sigma={sigma} has {len(cell)} results, "
                      f"expected {len(images) * 3}")
            psnr_mean, psnr_std = stats(cell, "psnr")
            ssim_mean, ssim_std = stats(cell, "ssim")
            time_mean, time_std = stats(cell, "runtime_s")
            table.append({"method": method, "sigma": sigma, "n": len(cell),
                          "psnr_mean": psnr_mean, "psnr_std": psnr_std,
                          "ssim_mean": ssim_mean, "ssim_std": ssim_std,
                          "runtime_mean": time_mean, "runtime_std": time_std,
                          "device": cell[0]["device"]})
    return table


def per_image(rows, images):
    """One dict per (method, sigma, image): mean and std over the seeds."""
    table = []
    for sigma in SIGMAS:
        for method in METHODS:
            for image in images:
                cell = select(rows, method=method, sigma=sigma, image=image)
                psnr_mean, psnr_std = stats(cell, "psnr")
                ssim_mean, ssim_std = stats(cell, "ssim")
                table.append({"method": method, "sigma": sigma, "image": image,
                              "n": len(cell),
                              "psnr_mean": psnr_mean, "psnr_std": psnr_std,
                              "ssim_mean": ssim_mean, "ssim_std": ssim_std})
    return table


def save_csv(table, path):
    """Save a list of dicts as a CSV file (numbers rounded for reading)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        for entry in table:
            writer.writerow({key: round(value, 4) if isinstance(value, float) else value
                             for key, value in entry.items()})


def best_methods(table, metric):
    """For each sigma, the method with the highest mean (oracles excluded, they use the clean image)."""
    best = {}
    for sigma in SIGMAS:
        fair = [e for e in table if e["sigma"] == sigma and "oracle" not in e["method"]]
        best[sigma] = max(fair, key=lambda e: e[metric])["method"]
    return best


def table1_latex(table):
    """Table 1 as LaTeX: methods x (PSNR, SSIM, time) for every sigma."""
    best_psnr = best_methods(table, "psnr_mean")
    best_ssim = best_methods(table, "ssim_mean")

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  % TODO: write the caption in your own words. Mention: mean +- std over",
        "  % 4 images x 3 seeds, bold = best without the clean image, oracles pick",
        "  % their setting with the clean image, BM3D is given sigma and DIP is not,",
        "  % DIP time on a Colab T4 GPU, TV and BM3D time on the Colab CPU.",
        "  \\caption{TODO}",
        "  \\label{tab:table1}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{l" + "ccc" * len(SIGMAS) + "}",
        "    \\toprule",
    ]
    header = "    "
    rules = "    "
    for i, sigma in enumerate(SIGMAS):
        first = 2 + 3 * i
        header += f" & \\multicolumn{{3}}{{c}}{{$\\sigma={sigma}$}}"
        rules += f"\\cmidrule(lr){{{first}-{first + 2}}}"
    lines += [header + " \\\\", rules,
              "    Method" + " & PSNR (dB) & SSIM & Time (s)" * len(SIGMAS) + " \\\\",
              "    \\midrule"]

    for method in METHODS:
        line = f"    {method}"
        for sigma in SIGMAS:
            e = [x for x in table if x["method"] == method and x["sigma"] == sigma][0]
            psnr = f"{e['psnr_mean']:.2f} $\\pm$ {e['psnr_std']:.2f}"
            ssim = f"{e['ssim_mean']:.3f} $\\pm$ {e['ssim_std']:.3f}"
            if best_psnr[sigma] == method:
                psnr = f"\\textbf{{{psnr}}}"
            if best_ssim[sigma] == method:
                ssim = f"\\textbf{{{ssim}}}"
            # "<" must be in math mode in LaTeX, or it prints as an upside-down "!".
            time = fmt_time(e["runtime_mean"]).replace("<", "$<$")
            line += f" & {psnr} & {ssim} & {time}"
        lines.append(line + " \\\\")

    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def per_image_latex(table, images):
    """Appendix table as LaTeX: PSNR / SSIM per method and image, one block per sigma."""
    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        "  % TODO: write the caption in your own words (mean over 3 seeds per image).",
        "  \\caption{TODO}",
        "  \\label{tab:per_image}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{ll" + "c" * len(images) + "}",
        "    \\toprule",
        "    $\\sigma$ & Method & " + " & ".join(images) + " \\\\",
    ]
    for sigma in SIGMAS:
        lines.append("    \\midrule")
        for method in METHODS:
            cells = []
            for image in images:
                e = [x for x in table if x["method"] == method
                     and x["sigma"] == sigma and x["image"] == image][0]
                cells.append(f"{e['psnr_mean']:.2f} / {e['ssim_mean']:.3f}")
            lines.append(f"    {sigma} & {method} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def print_table1(table):
    """Print Table 1 as plain text, one block per sigma."""
    for sigma in SIGMAS:
        print(f"\nsigma = {sigma}")
        print(f"  {'method':14s} {'PSNR (dB)':>15s} {'SSIM':>15s} {'time (s)':>9s}")
        for e in table:
            if e["sigma"] == sigma:
                print(f"  {e['method']:14s} "
                      f"{e['psnr_mean']:7.2f} +- {e['psnr_std']:4.2f} "
                      f"{e['ssim_mean']:7.3f} +- {e['ssim_std']:5.3f} "
                      f"{fmt_time(e['runtime_mean']):>9s}")


def main():
    """Read the results, then write and print the tables."""
    parser = argparse.ArgumentParser(description="Build Table 1")
    parser.add_argument("--in-dir", default="results/table1")
    args = parser.parse_args()

    rows = load_rows(os.path.join(args.in_dir, "metrics"))
    if not rows:
        raise SystemExit(f"no results found in {args.in_dir}/metrics")
    images = sorted({r["image"] for r in rows})
    print(f"{len(rows)} results, images: {', '.join(images)}")

    table = summary(rows, images)
    image_table = per_image(rows, images)

    save_csv(table, os.path.join(args.in_dir, "table1.csv"))
    save_csv(image_table, os.path.join(args.in_dir, "per_image.csv"))
    with open(os.path.join(args.in_dir, "table1.tex"), "w") as f:
        f.write(table1_latex(table))
    with open(os.path.join(args.in_dir, "per_image.tex"), "w") as f:
        f.write(per_image_latex(image_table, images))

    print_table1(table)
    print(f"\nsaved table1.csv, table1.tex, per_image.csv, per_image.tex in {args.in_dir}/")


if __name__ == "__main__":
    main()
