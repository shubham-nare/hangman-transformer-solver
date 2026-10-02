# Moving this project to another machine

> Written mid-competition on 2 September 2026, when run 4 (63.68% held-out) was
> the best model. Test counts, scores and the recommended run below reflect that
> point in time; see `README.md` for the final pipeline.

The whole project is ~157 MB and has no build step. Copy it, install four
packages, run the tests, resume training.

## 1. Move the files

This folder already lives inside OneDrive
(`C:\Users\Shubham\OneDrive\Documents\Claude Projects\Meltwater_Hackathon`), so
if the other laptop signs into the **same OneDrive account** it will sync on its
own. Just wait for the sync to finish and skip to step 2.

Otherwise copy the folder over (USB, network share, whatever). What matters:

| path | size | needed? |
|---|---|---|
| `hangman/`, `scripts/`, `tests/` | 0.6 MB | **yes** — all the code |
| `CONTEXT.md`, `EXPERIMENTS.md`, `README.md` | small | **yes** — decisions and results so far |
| `data/` | 15 MB | yes, or re-download (below) |
| `artifacts/run4/best_model.pt` | 36 MB | **yes** — the best model, 63.68% held-out |
| `submission.csv` | 4.6 MB | yes — best submission, 65.44% |
| `artifacts/run1,2,3,exp_*` | ~100 MB | **no** — superseded; their results are already in `EXPERIMENTS.md` |

Re-downloading the data instead of copying it:

```powershell
kaggle competitions download -c brand-buzzword-hackathon -p data
Expand-Archive data/brand-buzzword-hackathon.zip -DestinationPath data
```

## 2. Install

Python 3.10 or newer (this machine used 3.14.6).

```powershell
pip install numpy pandas pytest kaggle
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Match the CUDA build to the new GPU's driver — `cu126` works for most current
NVIDIA cards. Check the driver with `nvidia-smi`.

## 3. Verify before training

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q          # expect 78 passed
```

If the tests pass, the engine, sampler, model, and notebook builder are all
intact. Do not start a long run before this passes — a silent break in state
generation still trains happily and just scores worse.

## 4. Pick a config for the new GPU

Measured on the old 4 GB RTX 3050: d384/6L at batch 768 ran 3.3 steps/s and used
~3.2 GB. Scale from there. Peak memory is roughly linear in batch size and grows
with the square of `d_model`.

| VRAM | suggested | approx. batch |
|---|---|---|
| 8 GB | `--d-model 384 --n-layers 6 --dim-feedforward 1152` | `--batch-size 1536` |
| 12 GB | `--d-model 448 --n-layers 8 --dim-feedforward 1344` | `--batch-size 2048` |
| 16 GB+ | `--d-model 512 --n-layers 8 --dim-feedforward 1536` | `--batch-size 3072` |

Raise the learning rate a little with the batch size (we used 3.5e-4 at 768,
4.5e-4 at 2048). Benchmark before committing hours:

```powershell
python -m scripts.train_model --steps 200 --batch-size 2048 --d-model 448 --n-layers 8 --output-dir artifacts/probe
```

Watch `nvidia-smi` during it. If peak VRAM is above ~85% of the card, drop the
batch — going over does not error, it silently thrashes the allocator and costs
~30x throughput.

## 5. The run worth doing

The single clearest upside: **finish the 55k schedule.** Run 4 reached 63.68%
held-out (65.44% on the test set) at step 30,000 of 55,000 and was still
climbing when it was interrupted. Models gain most during the final learning-rate
decay, which it never got.

```powershell
python -u -m scripts.train_model `
  --steps 55000 --batch-size 768 --learning-rate 3.5e-4 `
  --d-model 384 --n-layers 6 --n-heads 8 --dim-feedforward 1152 `
  --eval-every 2500 --eval-words 5000 --resume `
  --output-dir artifacts/run5
```

Scale batch/model per the table above for a bigger card. `--resume` picks up from
`last_state.pt` if the run dies, so an interruption costs one eval interval
rather than the whole run.

Then:

```powershell
python -m scripts.generate_submission --checkpoint artifacts/run5/best_model.pt
python -m scripts.compare_models --checkpoints artifacts/run4/best_model.pt artifacts/run5/best_model.pt --ensemble
```

## 6. Do not retry these

Measured, negative, and already paid for. Details in `EXPERIMENTS.md`.

| change | result |
|---|---|
| DAgger self-play at 50% of batch | **−3.1** |
| binary presence objective | **−2.6** |
| EMA weight averaging | +0.14 (noise) |
| equal-weight ensemble of unequal models | −0.3 |

## 7. Kaggle CLI (optional)

Only needed to re-download data, push notebooks, or submit.

```powershell
kaggle auth login          # browser OAuth; the token is short-lived
```

Note the leaderboard endpoint is not covered by the OAuth scope the CLI gets —
read standings in a browser. Kernel runs are drivable headlessly:
`kaggle kernels push -p kaggle_kernel --accelerator nvidiaTeslaT4x2`.
