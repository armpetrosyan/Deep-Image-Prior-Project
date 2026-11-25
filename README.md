# Deep Image Prior: images are fitted faster than noise

University project for DS330 Deep Learning. We verify one claim from the paper
"Deep Image Prior" (Ulyanov, Vedaldi, Lempitsky, IJCV): an untrained
convolutional network fitted to a single image fits natural images much faster
than noise, so stopping the fit early denoises the image. The project covers
denoising only.

## Files

- `utils.py`: loading and saving images, noisy / shuffled / noise targets, and the metrics (MSE, PSNR, SSIM).
- `models.py`: the paper's "skip" hourglass network.
- `dip.py`: the DIP fitting loop (`run_dip`) and saving/loading of results.
- `smoke_test.py`: an end-to-end check of the whole pipeline, plus a time estimate for the full runs.
- `images/`: the test images shared by the team.

Outputs are written to `results/`, which is not committed.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Checks

```
python smoke_test.py   # end-to-end test (uses a small, fast mode when there is no GPU)
python utils.py        # self-check of each file
python models.py
python dip.py
```

## Running the experiments

The real runs (256x256 images, 3000 iterations) are too slow on a CPU. Run them
on Google Colab with a GPU. The code works on both CPU and CUDA.
