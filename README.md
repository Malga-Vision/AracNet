# AracNet: Revealing Debiasing Signals across Layers with Shallow Monitors

**ECCV 2026** — Vito Paolo Pastore, Massimiliano Ciranni, Enzo Tartaglione, Vittorio Murino

> Deep neural networks often show poor generalization when trained on biased datasets presenting spurious relations between samples and target labels. AracNet is an unsupervised debiasing framework that replaces the *when to stop* paradigm with a *where to look* one. By attaching one linear classifier (Shallow Monitor) to each block of a pre-trained network, AracNet automatically identifies the layer where bias is most separable and uses it to reweight samples during a self-mitigation phase — without requiring bias labels, annotated validation sets, or heuristic early stopping.

---

## Requirements

```bash
pip install torch torchvision tqdm numpy matplotlib seaborn wandb requests gdown
```

A CUDA-capable GPU is required. All experiments were run on an NVIDIA A100 (20 GB VRAM).

---

## Running experiments

Datasets are downloaded automatically on first run under `./data/`. Each run trains a seed-specific biased backbone (Phase 1) then runs the full AracNet debiasing pipeline (Phase 2). Results are logged via Weights & Biases and checkpoints are saved under `./hist/SIMPLE/<dataset>/`.

### Waterbirds

```bash
bash Waterbirds.sh
```

Equivalent single-seed invocation:

```bash
python main_aracnet.py \
    --dataset waterbirds \
    --epochs_base_model 50 \
    --epochs_debiasing 50 \
    --seed 0 \
    --rho 95 \
    --train_body_aracnet
```

### BFFHQ

```bash
bash BFFHQ.sh
```

Equivalent single-seed invocation:

```bash
python main_aracnet.py \
    --dataset BFFHQ \
    --epochs_base_model 50 \
    --epochs_debiasing 70 \
    --seed 0 \
    --rho 99.5 \
    --train_body_aracnet
```

### CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name (`waterbirds`, `BFFHQ`) | required |
| `--epochs_base_model` | Phase 1 training epochs | `100` |
| `--epochs_debiasing` | Phase 2 debiasing epochs | `50` |
| `--seed` | Random seed | `0` |
| `--rho` | Bias correlation degree (e.g. `95` → ρ = 0.95) | `95` |
| `--train_body_aracnet` | Flag — run Phase 1 backbone training | off |

> **Note:** omitting `--train_body_aracnet` skips Phase 1 and loads an existing backbone checkpoint for the given seed. The checkpoint must have been produced by a prior run with the same `--seed`, `--dataset`, and `--rho`.

---

## Checkpoints

Backbone and debiased model checkpoints are saved per-seed under:

```
./hist/SIMPLE/<dataset>/<rho>/<backbone>/
    aracnet-<backbone>-False_<dataset>_<rho>_seed<seed>-biased-final.pth   # biased backbone
    debiased_<dataset>_<rho>_seed<seed>.pth                                 # debiased model
```

---

## Citation

<!-- ```bibtex
@inproceedings{pastore2026aracnet,
  title     = {AracNet: Revealing Debiasing Signals across Layers with Shallow Monitors},
  author    = {Pastore, Vito Paolo and Ciranni, Massimiliano and Tartaglione, Enzo and Murino, Vittorio},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
``` -->
