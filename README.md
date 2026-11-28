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

## Running on Colab

1. Open `colab_run.ipynb` in Colab (File -> Open notebook -> GitHub, and paste the repo URL).
2. Choose a GPU: Runtime -> Change runtime type -> T4 GPU.
3. Run the cells from top to bottom. The notebook checks the GPU, mounts Google
   Drive, clones or updates the repo, installs the missing packages and runs the
   smoke test.
4. `results/` is linked to `MyDrive/DIP_results` on your Drive, so results are
   kept when the runtime disconnects.
5. Put your own script in the last cell, e.g. `!python exp_fitting.py`.

After a disconnect, run all the cells again: the code is pulled again and
`results/` is linked to Drive again.

## Results

Final results are in `final_results/`, one folder per part.

- `partA_fig1_fitting_speed/`: Fig 1 (`figures/`) and its tables (`tables/`).
  Made with `python exp_fitting.py` then `python plot_fig1.py`
  (48 runs on a Colab T4; code at commit e2ad139).
- `partB_table1_dip_vs_baselines/`: Table 1 and the per-image table (`tables/`)
  and the tuning plot (`figures/`). Made with `python tune.py`, then
  `python exp_table1.py` and `python make_table1.py`
  (36 cases; DIP on a Colab T4, TV and BM3D on the Colab CPU; code at commit e2ad139).
