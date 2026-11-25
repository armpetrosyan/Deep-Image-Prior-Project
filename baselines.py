"""Training-free denoisers that DIP is compared against (Table 1).

Every function takes a noisy image x0 as a [3, H, W] tensor in [0, 1] and
returns the denoised image in the same format (on the CPU).
"""

import time

import bm3d
import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

import utils


def no_prior(x0):
    """No denoising at all: the noisy image itself (the lower bound)."""
    return x0.detach().cpu().clone()


def tv_denoise(x0, weight):
    """Total-variation denoising (Chambolle). Larger weight = smoother output."""
    array = utils.to_numpy(x0)   # [H, W, 3]
    # channel_axis=-1: treat the last axis as RGB, not as a 3-pixel-wide image.
    out = denoise_tv_chambolle(array, weight=weight, channel_axis=-1)
    return to_tensor(out)


def bm3d_denoise(x0, sigma):
    """BM3D for colour images. Unlike DIP and TV, it is told the noise level."""
    array = utils.to_numpy(x0)   # [H, W, 3]
    # bm3d expects the noise std on the same [0, 1] scale as the image.
    out = bm3d.bm3d_rgb(array, sigma_psd=sigma / 255.0)
    # BM3D can go slightly outside [0, 1]; DIP's output cannot (Sigmoid), so clip.
    out = np.clip(out, 0.0, 1.0)
    return to_tensor(out)


def to_tensor(array):
    """Convert a [H, W, 3] numpy array to a float32 [3, H, W] tensor."""
    array = np.asarray(array, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def timed(function, *args, **kwargs):
    """Call function(*args, **kwargs) and return (result, runtime in seconds)."""
    start = time.time()
    result = function(*args, **kwargs)
    return result, time.time() - start


if __name__ == "__main__":
    # Quick self-check on one test image at sigma = 25 (runs on CPU in seconds).
    clean = utils.load_image("images/parrots.png", size=256)
    sigma = 25
    noisy = utils.add_noise(clean, sigma=sigma, seed=0)

    methods = {
        "no prior": lambda: no_prior(noisy),
        "TV (w=0.08)": lambda: tv_denoise(noisy, weight=0.08),
        "BM3D": lambda: bm3d_denoise(noisy, sigma=sigma),
    }
    for name, method in methods.items():
        out, runtime = timed(method)
        in_range = out.min().item() >= 0 and out.max().item() <= 1
        print(f"{name:12s} PSNR {utils.psnr(out, clean):6.2f} dB  "
              f"SSIM {utils.ssim(out, clean):.4f}  {runtime:6.2f} s  "
              f"shape {tuple(out.shape)} dtype {out.dtype} in [0, 1]: {in_range}")
