# DISCO

Reference implementation for the DISCO anomaly-detection method. DISCO learns from a
**mixed (partly-contaminated) training set** by decomposing it as

```
X = L + S + E
```

* **L** — low-rank background, reconstructed by a convolutional autoencoder whose latent
  *positional matrix* `P` is nuclear-norm penalized (sample-wise low-rank prior);
* **S** — sparse anomaly (soft-thresholding);
* **E** — dense noise (ridge / L2 closed form).

The decomposition is solved by an ADMM loop that alternates between updating `L`, the
closed-form `S`/`E`/dual variables, and the autoencoder weights. Anomalies are then read off
the recovered sparse component `S`.

---

## Computer and software environment

Results in the paper were produced with **PyTorch 2.0.1 (CUDA 11.8 build)** in **Python 3.10.6**
on **Windows 11 (64-bit)**, running on an **NVIDIA GeForce RTX 3060 Laptop GPU** (6 GB, compute
capability 8.6; CUDA 11.8, cuDNN 8.7) and an **AMD Ryzen 9 5900HS with Radeon Graphics (3.30 GHz)**.

## Installation

1. Install Python 3.10 (tested on 3.10.6); optionally create a fresh environment
   (`conda create -n disco python=3.10 && conda activate disco`).
2. Install the dependencies (this pulls the CUDA-11.8 PyTorch build from the PyTorch index):
   ```bash
   pip install -r requirements.txt
   ```
   For a CPU-only or different-CUDA machine, swap the two `+cu118` pins in `requirements.txt`
   for the matching PyTorch 2.0.1 build. No compilation step is required.

## Folder structure

```
Codes/
├── models.py              network definitions: LRAE_2d (DISCO autoencoder), MemAE_2d
├── utlis.py               MVTEC dataset loader, ADMM operators (SoftThresholding),
│                          RPCA_gpu, and metrics (Dice, PSNR, SSIM, DMS)
├── train_RobMemAE.py      DISCO training — pretraining + ADMM (X = L + S + E);
│                          also the rpca / memae baselines
├── test_RobMemAE.py       evaluation: Dice + DMS-PSNR / DMS-SSIM (background restoration)
├── roc_pixelevel.py       evaluation: threshold-free pixel-level AUROC / AUPRC (+ combined)
├── train_rdae.py          RDAE bottleneck-variant baseline — training
├── test_rdae.py           RDAE bottleneck-variant baseline — evaluation
├── simulation_data.py     builds the simulated contaminated datasets (blobs + lines)
├── check_rank.py          latent positional-matrix rank / singular-value analysis
├── plot_dms.py            DMS-metric figure script
├── plot_sensitivity.py    hyperparameter-sensitivity figure script
├── requirements.txt
├── README.md
├── LICENSE
│
├── DRAEM/                 DRÆM baseline (Zavrtanik et al., ICCV 2021; MIT © VitjanZ) —
│                          original code with minor edits to train_DRAEM.py / test_DRAEM.py
│                          for this folder layout; its checkpoints/ & datasets/ are external
│
├── data/                              INPUT DATA
│   ├── BTech_Dataset_Transformed/     real BTAD images — case study (category BTAD3_sys)
│   └── btad_simulation/               simulated data, one folder per contamination level
│       └── 02_sim_contam_level_0.02/  (also _0.05 / _0.08 / _0.1)
│           └── <category>/train/      each category (e.g. 02_sim1/2/3) has:
│               ├── good/                  the mixed training set (X)
│               ├── defect/               contaminated samples (evaluated)
│               ├── defect_background/    clean background   (L ground truth)
│               └── defect_ground_truth/  binary anomaly mask (S ground truth)
│
├── temp_var/                          TRAINED ARTIFACTS (evaluation loads from here)
│   ├── <category>/                    weights (.pth) + decomposition tensors L/S/E/Y (.pt);
│   │                                  the epoch to load is set in the `model_para` dict
│   │                                  inside test_RobMemAE.py / roc_pixelevel.py
│   ├── pretrained_weight/             autoencoder pretraining checkpoints
│   └── rdae_bottleneck_variants/      RDAE variant artifacts
│
└── simulation/                        intermediate products of simulation_data.py
    └── btad2/                         low-rank backgrounds + per-contamination anomaly sets
```

## Data & weights

The datasets and trained artifacts are too large for version control and are hosted externally:

**➜ [Data & weights (Google Drive)](https://drive.google.com/drive/folders/1oFihSWU6XCDnWORm-I7RtUCFmF49TKvE?usp=sharing)**

Download and place the folders at the paths shown in the structure above before running:
`data/` (BTAD + simulation images), `temp_var/` (weights + `L/S/E/Y` decomposition tensors +
pretrained checkpoints), `simulation/` (intermediate simulation products), and the DRÆM baseline's
`DRAEM/checkpoints/` and `DRAEM/datasets/` (including the DTD anomaly-source textures).

## Running

Each script is configured by editing the variables at the top (`dataset`, `category`,
`model_running`), then run directly — there is no command-line interface.

```bash
# 1. (optional) regenerate the simulated datasets
python simulation_data.py

# 2. train DISCO (set dataset / category / model_running at the top of the file)
python train_RobMemAE.py

# 3. evaluate
python test_RobMemAE.py     # Dice, DMS-PSNR, DMS-SSIM
python roc_pixelevel.py     # pixel-level AUROC / AUPRC (+ simulation-combined)
```

`model_running` selects the method: `disco`, the ablations `disco-wo-p` / `disco-wo-e` /
`disco-wo-ep`, or the baselines `rdae` / `rpca` / `memae` (and `draem`). The RDAE
bottleneck-variant study uses `train_rdae.py` / `test_rdae.py`.

The **DRÆM** baseline is in [`DRAEM/`](DRAEM/) — the original DRÆM implementation (Zavrtanik
et al., ICCV 2021; MIT © VitjanZ, see `DRAEM/LICENSE`) with minor edits to `train_DRAEM.py` /
`test_DRAEM.py` for this folder layout. It is trained/evaluated on its own; its output anomaly
map is saved as `temp_var/<category>/S_<category>_draem.pt`, which the DISCO evaluation scripts
then load like any other method. Its `checkpoints/` and `datasets/` are hosted externally.

## Reproducing the reported numbers

The evaluation scripts (`test_RobMemAE.py`, `roc_pixelevel.py`) define an in-script `model_para`
dictionary mapping each `(category, model)` to `[training_epoch, threshold]`. These select the
corresponding weights and `L/S/E` tensors under `temp_var/<category>/` and reproduce the paper's results — e.g. DISCO on the simulation
data gives pooled AUROC 0.983 / 0.997 / 0.988 (sim1/2/3) and 0.988 combined. Set
`category` and `model_running` to the desired run and execute the evaluation scripts above.

## Reproducing the paper's figures and tables

| Result | Script(s) | Mode / settings |
|---|---|---|
| Fig 2 — DMS metric illustration | `plot_dms.py` (reads `temp_var/DMS5.xlsx`) | plots DMS₅-PSNR/SSIM curves |
| Figs 3, 6 — latent-rank comparison | `check_rank.py` | `category='02_sim1'` (Fig 3) / `'BTAD3_sys'` (Fig 6); `variants=['disco','disco-wo-p']` |
| Figs 4, 7, 8, 9 — qualitative panels | `test_RobMemAE.py` / `test_rdae.py` | `whether_plot=True`, `whether_just_result=False`; one run per method/variant |
| Fig 5 — sensitivity plots | `plot_sensitivity.py` | plots the values of Tables 3–5 |
| Tables 1, 6 — threshold-free (sim / case study) | `roc_pixelevel.py` | `eval_contam_levels=False`, `eval_rdae_variants=False` |
| Table 3 — threshold-free sensitivity | `roc_pixelevel.py` | `eval_contam_levels=True` for the 5/8/10% columns |
| Tables 2, 4, 5, 7, 8, 9 — threshold-dependent (Dice + DMS₂₀) | `test_RobMemAE.py` / `test_rdae.py` | `whether_plot=True`; Dice + DMS₂₀ together. Tables **2, 7, 8, 9** (blob/line or per-type breakdown) also need `whether_just_result=False`; Tables **4, 5** (overall only) use the default (`True`). |
| Sec 4.4 — RDAE bottleneck search | `train_rdae.py` → `test_rdae.py` / `roc_pixelevel.py` (`eval_rdae_variants=True`) | 5 stride configs; `s234` (5×5) is best |
| DRÆM baseline (all DRÆM entries) | `DRAEM/train_DRAEM.py` → `DRAEM/test_DRAEM.py`; then `roc_pixelevel.py` | `test_DRAEM.py` does DRÆM's Dice/DMS (Tables 2, 4, 5, 7) + panels (Figs 4, 7); `roc_pixelevel.py` loads its exported map for AUROC/AUPRC (Tables 1, 3, 6) |

**Notes on the workflow.**
1. In all **benchmark** tables/figures, the **RDAE** results are the best bottleneck variant
   (`s234`, 5×5 latent), reproduced via `test_rdae.py` / `roc_pixelevel.py` with
   `eval_rdae_variants=True` — **not** the generic `rdae` output of the main scripts.
2. Several results **share one execution**: the threshold-dependent metrics (Tables 2, 4, 5, 7,
   8, 9) come from `test_RobMemAE.py` / `test_rdae.py`, which emit Dice and DMS₂₀-PSNR/SSIM in a
   single pass (DRÆM excepted — see note 5); the threshold-free metrics (Tables 1, 3, 6) come
   from `roc_pixelevel.py`; and Figure 5 plots the same numbers as Tables 3–5.
3. The **2%-anomaly-ratio** column of the sensitivity tables equals the simulation-study values
   in Tables 1, 2 and 6.
4. The case study (`BTAD3_sys`) is **detection-only** — real BTAD data has no background ground
   truth, so it has no DMS/restoration results (Figs 7/9 show only the restored background).
5. **DRÆM is a self-contained baseline**, trained and evaluated in `DRAEM/` (`train_DRAEM.py` →
   `test_DRAEM.py`). `test_DRAEM.py` computes DRÆM's Dice + DMS₂₀-PSNR/SSIM (Tables 2, 4, 5, 7)
   and its qualitative panels (Figures 4, 7) itself; it also exports
   `temp_var/<category>/S_<category>_draem.pt`, which **`roc_pixelevel.py`** loads to compute
   DRÆM's threshold-free AUROC/AUPRC (Tables 1, 3, 6). `test_RobMemAE.py` does **not** handle
   DRÆM (see the *DRÆM baseline* section below).
6. `test_RobMemAE.py`'s `whether_just_result=True` prints only the **overall** Dice / PSNR / SSIM
   summary; set it **`False`** to also print the **group breakdown** (blob/line for simulation,
   per-defect-type for the case study) and the per-sample panels. So Tables **2, 7, 8, 9** and
   the qualitative figures need `whether_just_result=False`, while Tables **4, 5** (overall only)
   use the default `True`.
7. The **DISCO** scripts are configured by editing variables at the top (`dataset` / `category` /
   `model_running`); the **DRÆM** scripts instead take command-line arguments (see
   `DRAEM/command.txt`). Run times refer to the hardware in the *Computer and software
   environment* section.

### DRÆM baseline

DRÆM ([`DRAEM/`](DRAEM/)) is trained and evaluated in its own folder. It computes its own Dice /
DMS metrics and qualitative panels; only its threshold-free AUROC/AUPRC is computed by the DISCO
pipeline, from an exported anomaly map.

1. **Train + evaluate** (run inside `DRAEM/`):
   ```bash
   # train one model per category (--obj_id selects the category)
   python train_DRAEM.py --gpu_id 0 --obj_id 6 --lr 0.0001 --bs 8 --epochs 500 \
       --data_path ./datasets/data/ --anomaly_source_path ./datasets/dtd/images/ \
       --checkpoint_path ./checkpoints/ --log_path ./logs/
   # evaluate + export the anomaly map
   python test_DRAEM.py --gpu_id 0 --obj_id 5 --base_model_name "DRAEM_test_0.0001_500_bs8" \
       --data_path ./datasets/data/ --checkpoint_path ./checkpoints/
   ```
   `test_DRAEM.py` computes DRÆM's Dice + DMS₂₀-PSNR/SSIM (**Tables 2, 4, 5, 7**) and its
   qualitative panels (**Figures 4, 7**) directly. Its `save_train_S` also writes
   `S_defect = X − reconstruction` to `temp_var/<category>/S_<category>_draem.pt`, indexed to
   match the MVTEC loader.

2. **Threshold-free metrics** — `roc_pixelevel.py` (with `model_running='draem'`) loads
   `S_<category>_draem.pt` and computes DRÆM's AUROC/AUPRC (**Tables 1, 3, 6**) in the same pass
   as the other methods. (`test_RobMemAE.py` does not evaluate DRÆM.)

Training needs the external **Describable Textures Dataset (DTD)** as the anomaly source
(`--anomaly_source_path`); `DRAEM/scripts/download_dataset.sh` fetches it. DRÆM's `checkpoints/`
and `datasets/` are hosted externally (like `data/` and `temp_var/`, see the note above).

## License

Released under the MIT License — see [`LICENSE`](LICENSE). Copyright (c) 2026 Wang, Mou, Shi, Zhang.
The bundled DRÆM baseline in [`DRAEM/`](DRAEM/) retains its own MIT License (© 2021 VitjanZ,
`DRAEM/LICENSE`).
