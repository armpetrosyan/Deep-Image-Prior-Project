"""Deep Image Prior fitting loop (paper Eq. 4).

An untrained SkipNet f(z) with a fixed random input z is fitted to one target
image by minimising mean((f(z) - target)^2) over the network weights.
"""

import json
import os
import time

import numpy as np
import torch

import models
import utils


def run_dip(target, clean=None, iters=3000, seed=0, net_cfg=None, lr=0.01,
            log_every=10, save_at=(), input_noise=0.0, ema=None, device=None,
            save_path=None, verbose=True):
    """Fit a new SkipNet to `target` and return a dict with the results."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    if device == "cuda":
        # Deterministic cuDNN convolutions so GPU runs are (nearly) exactly repeatable.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if net_cfg is None:
        net_cfg = {}

    height = target.shape[1]
    width = target.shape[2]
    target = target.to(device).unsqueeze(0)   # add a batch dimension: [1, 3, H, W]
    if clean is not None:
        clean = clean.to(device)

    # Same seed -> same starting weights and same z, whatever the target is.
    torch.manual_seed(seed)
    net = models.skip_net(**net_cfg).to(device)
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(1, 32, height, width, generator=generator) * 0.1
    z = z.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    losses = []
    psnrs = []
    psnrs_ema = []
    psnr_iters = []
    best_psnr = None
    best_iter = None
    best_img = None
    out_avg = None
    imgs_at = {}
    # The same, but for the moving-average output (only used when ema is set).
    best_ema_psnr = None
    best_ema_iter = None
    best_ema_img = None
    imgs_at_ema = {}

    start_time = time.time()
    for i in range(1, iters + 1):
        # Optionally perturb z a little each step (a regulariser from the paper).
        if input_noise > 0:
            z_input = z + input_noise * torch.randn_like(z)
        else:
            z_input = z

        optimizer.zero_grad()
        output = net(z_input)
        loss = torch.mean((output - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        image = output.detach()[0]   # drop the batch dimension: [3, H, W]

        # Optional exponential moving average of the outputs (smoother result).
        if ema is not None:
            if out_avg is None:
                out_avg = image
            else:
                out_avg = ema * out_avg + (1 - ema) * image

        # Track PSNR against the clean image (used only for measuring).
        if clean is not None and i % log_every == 0:
            current_psnr = utils.psnr(image, clean)
            psnrs.append(current_psnr)
            psnr_iters.append(i)
            if ema is not None:
                ema_psnr = utils.psnr(out_avg, clean)
                psnrs_ema.append(ema_psnr)
                if best_ema_psnr is None or ema_psnr > best_ema_psnr:
                    best_ema_psnr = ema_psnr
                    best_ema_iter = i
                    best_ema_img = out_avg.cpu()
            # Remember the best output so far ("oracle" early stopping).
            if best_psnr is None or current_psnr > best_psnr:
                best_psnr = current_psnr
                best_iter = i
                best_img = image.cpu()

        if i in save_at:
            imgs_at[i] = image.cpu()
            if out_avg is not None:
                imgs_at_ema[i] = out_avg.cpu()

        if verbose and (i % 500 == 0 or i == iters):
            message = f"iter {i:5d}  loss {losses[-1]:.5f}"
            if clean is not None and i % log_every == 0:
                message += f"  PSNR {psnrs[-1]:.2f} dB"
            print(message)
    runtime = time.time() - start_time

    ema_img = None
    if out_avg is not None:
        ema_img = out_avg.cpu()

    result = {
        "loss": np.array(losses),
        "psnr": np.array(psnrs),
        "psnr_ema": np.array(psnrs_ema),
        "psnr_iters": np.array(psnr_iters),
        "best_iter": best_iter,
        "best_psnr": best_psnr,
        "best_img": best_img,
        "final_img": image.cpu(),
        "ema_img": ema_img,
        "best_ema_iter": best_ema_iter,
        "best_ema_psnr": best_ema_psnr,
        "best_ema_img": best_ema_img,
        "imgs_at": imgs_at,
        "imgs_at_ema": imgs_at_ema,
        "runtime_s": runtime,
        "seed": seed,
        "iters": iters,
        "net_cfg": net_cfg,
    }
    if save_path is not None:
        save_result(result, save_path)
    return result


def save_result(result, path):
    """Save a run_dip result to a compressed .npz file (images as [H, W, 3])."""
    data = {}
    for key, value in result.items():
        if value is None:
            continue
        if key == "imgs_at":
            # One entry per saved iteration, e.g. "img_at_1800".
            for iteration, img in value.items():
                data[f"img_at_{iteration}"] = utils.to_numpy(img)
        elif key == "imgs_at_ema":
            # The same for the moving-average output, e.g. "ema_img_at_1800".
            for (iteration,
                 img) in value.items():
                data[f"ema_img_at_{iteration}"] = utils.to_numpy(img)
        elif key == "net_cfg":
            # np.savez cannot store a dict without pickle, so store it as text.
            data[key] = json.dumps(value)
        elif torch.is_tensor(value):
            data[key] = utils.to_numpy(value)
        else:
            data[key] = np.asarray(value)

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez_compressed(path, **data)


def load_result(path):
    """Load a saved result as a dict of NumPy arrays."""
    result = {}
    with np.load(path) as f:
        for key in f.files:
            result[key] = f[key]
    return result


if __name__ == "__main__":
    # Quick self-check on a small image (runs on CPU in about a minute).
    clean_image = utils.load_image("images/parrots.png", size=64)
    noisy_image = utils.add_noise(clean_image, sigma=25, seed=0)
    print(f"noisy input PSNR: {utils.psnr(noisy_image, clean_image):.2f} dB")

    result = run_dip(noisy_image, clean_image, iters=150, seed=0,
                     log_every=10, save_at=(100,))
    first_loss = result["loss"][0]
    last_loss = result["loss"][-1]
    print(f"first loss {first_loss:.5f}, last loss {last_loss:.5f}, "
          f"went down: {last_loss < first_loss}")
    print(f"best PSNR {result['best_psnr']:.2f} dB at iter {result['best_iter']}, "
          f"runtime {result['runtime_s']:.1f} s")

    # Same seed again: the first losses must be exactly the same.
    short_result = run_dip(noisy_image, clean_image, iters=5, seed=0, verbose=False)
    same = np.array_equal(result["loss"][:5], short_result["loss"])
    print("same seed -> identical first 5 losses:", same)

    save_result(result, "results/selfcheck.npz")
    loaded = load_result("results/selfcheck.npz")
    print("saved keys:", sorted(loaded.keys()))
    print("best_img shape:", loaded["best_img"].shape,
          "loss matches:", np.array_equal(loaded["loss"], result["loss"]))
