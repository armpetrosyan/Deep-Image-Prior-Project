"""Image loading/saving, noisy targets and quality metrics.

Images are float32 tensors of shape [3, H, W] with values in [0, 1].
Noise sigma is given on the 0-255 scale (15, 25, 50).
"""

import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def load_image(path, size=256):
    """Load an RGB image, centre-crop it to a square and resize to size x size."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.BICUBIC)

    array = np.asarray(image, dtype=np.float32) / 255.0   # [H, W, 3]
    tensor = torch.from_numpy(array).permute(2, 0, 1)     # [3, H, W]
    return tensor.contiguous()


def save_image(x, path):
    """Save a [3, H, W] tensor as a PNG file."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    array = to_numpy(x.clamp(0, 1))
    array = np.round(array * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def to_numpy(x):
    """Convert a [3, H, W] tensor to a [H, W, 3] float32 numpy array."""
    return x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def add_noise(x, sigma, seed, clip=True):
    """Add Gaussian noise with std sigma/255 (sigma on the 0-255 scale)."""
    generator = torch.Generator().manual_seed(seed)
    # Draw the noise on the CPU so the same seed gives the same noise on any device.
    noise = torch.randn(x.shape, generator=generator) * (sigma / 255.0)
    noisy = x + noise.to(x.device)
    if clip:
        noisy = noisy.clamp(0, 1)
    return noisy


def shuffle_pixels(x, seed):
    """Randomly move pixels around, keeping each pixel's RGB values together."""
    channels, height, width = x.shape
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(height * width, generator=generator).to(x.device)
    pixels = x.reshape(channels, height * width)   # one column per pixel
    shuffled = pixels[:, order]
    return shuffled.reshape(channels, height, width)


def matched_noise(x, seed):
    """Gaussian noise with the same per-channel mean and std as x, clamped to [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    mean = x.mean(dim=(1, 2), keepdim=True)   # shape [3, 1, 1]
    std = x.std(dim=(1, 2), keepdim=True)
    noise = torch.randn(x.shape, generator=generator).to(x.device)
    # The network ends with a Sigmoid, so it can only output values in [0, 1].
    # Values outside that range could never be fitted and would make pure
    # noise look artificially slow to fit.
    return (mean + std * noise).clamp(0, 1)


def mse(a, b):
    """Mean squared error between two images."""
    return torch.mean((a - b) ** 2).item()


def psnr(a, b):
    """Peak signal-to-noise ratio in dB, for images in [0, 1]."""
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return 10 * np.log10(1.0 / error)


def ssim(a, b):
    """Structural similarity between two images (1 means identical)."""
    return structural_similarity(to_numpy(a), to_numpy(b),
                                 channel_axis=-1, data_range=1.0)


if __name__ == "__main__":
    # Quick self-check on a random image.
    x = torch.rand(3, 256, 256, generator=torch.Generator().manual_seed(0))

    noisy_1 = add_noise(x, 25, seed=1)
    noisy_2 = add_noise(x, 25, seed=1)
    print("add_noise reproducible:", torch.equal(noisy_1, noisy_2))

    # Without clipping the PSNR should be about 20.17 dB; clipping lowers the error.
    noisy_unclipped = add_noise(x, 25, seed=1, clip=False)
    print(f"PSNR sigma=25 (no clip): {psnr(x, noisy_unclipped):.2f} dB")
    print(f"PSNR sigma=25 (clip):    {psnr(x, noisy_1):.2f} dB")

    shuffled = shuffle_pixels(x, seed=2)
    values_before = torch.sort(x.flatten()).values
    values_after = torch.sort(shuffled.flatten()).values
    print("shuffle keeps pixel values:", torch.equal(values_before, values_after))

    print(f"SSIM(x, x) = {ssim(x, x):.4f}")

    noise = matched_noise(x, seed=3)
    low = noise.min().item()
    high = noise.max().item()
    print(f"matched_noise min={low:.4f} max={high:.4f} in [0, 1]:",
          low >= 0 and high <= 1)
