from models import LRAE_2d
from utlis import MVTEC
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

## ── Config ───────────────────────────────────────────────────────────────────
# '02_sim1' for simulation
# 'BTAD3_sys' for case study
im_shape     = 128
category     = 'BTAD3_sys'
variants     = ['disco', 'disco-wo-p']   # any keys present in model_para[category]
use_pretrain = False

## ── Auto-derived ─────────────────────────────────────────────────────────────
kernel_size  = 5
eta          = 0.999
fontsize_label = 14   # x/y axis labels
fontsize_title = 16   # subplot titles
fontsize_tick  = 12   # tick labels
fontsize_legend = 12  # legend

variant_labels = {
    'disco':        'DISCO',
    'disco-wo-p':   'DISCO-w/o-P',
    'disco-wo-e':   'DISCO-w/o-E',
    'disco-wo-ep':  'DISCO-w/o-EP',
}

data_path_map = {
    '02_sim1':                  'btad_simulation\\02_sim_contam_level_0.02',
    '02_sim2':                  'btad_simulation\\02_sim_contam_level_0.02',
    '02_sim3':                  'btad_simulation\\02_sim_contam_level_0.02',
    'BTAD3_sys':                'BTech_Dataset_Transformed',
    '02_sim_contam_level_0.05': 'btad_simulation',
    '02_sim_contam_level_0.08': 'btad_simulation',
    '02_sim_contam_level_0.1':  'btad_simulation',
}
batch_size_map = {
    '02_sim1':                  100,
    '02_sim2':                  100,
    '02_sim3':                  100,
    'BTAD3_sys':                184,
    '02_sim_contam_level_0.05': 100,
    '02_sim_contam_level_0.08': 100,
    '02_sim_contam_level_0.1':  100,
}
model_para = {
    '02_sim1':                   {'disco': 106, 'disco-wo-p': 105, 'disco-wo-e': 105, 'disco-wo-ep': 108},
    '02_sim2':                   {'disco': 104, 'disco-wo-p': 105, 'disco-wo-e': 105, 'disco-wo-ep': 108},
    '02_sim3':                   {'disco': 104, 'disco-wo-p': 105, 'disco-wo-e': 104, 'disco-wo-ep': 107},
    'BTAD3_sys':                 {'disco': 213, 'disco-wo-p': 209, 'disco-wo-e': 214, 'disco-wo-ep': 212},
    '02_sim_contam_level_0.05':  {'disco': 105},
    '02_sim_contam_level_0.08':  {'disco': 105},
    '02_sim_contam_level_0.1':   {'disco': 104},
}
pretrain_epoch_map = {
    '02_sim1':                  100,
    '02_sim2':                  100,
    '02_sim3':                  100,
    'BTAD3_sys':                200,
    '02_sim_contam_level_0.05': 100,
    '02_sim_contam_level_0.08': 100,
    '02_sim_contam_level_0.1':  100,
}
pretrain_name_map = {
    '02_sim1':                  '02_sim_1',
    '02_sim2':                  '02_sim_1',
    '02_sim3':                  '02_sim_1',
    'BTAD3_sys':                'BTAD3_sys',
    '02_sim_contam_level_0.05': '02_sim_1',
    '02_sim_contam_level_0.08': '02_sim_1',
    '02_sim_contam_level_0.1':  '02_sim_1',
}

batch_size   = batch_size_map[category]
root_path    = 'data\\' + data_path_map[category]
weight_label = 'pretrain' if use_pretrain else 'trained'
device       = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_weight_path(variant):
    if use_pretrain:
        epoch = pretrain_epoch_map[category]
        pname = pretrain_name_map[category]
        return f'temp_var\\pretrained_weight\\weight_{pname}_{kernel_size}_500.pth', epoch
    epoch = model_para[category][variant]
    return f'temp_var\\{category}\\weight_{category}_{variant}_{kernel_size}_{epoch}.pth', epoch


def load_model(variant):
    weight_path, epoch = get_weight_path(variant)
    model = LRAE_2d(chnum_in=3, kernel_size=kernel_size).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f'Using {device}  |  category: {category}  |  variant: {variant}  |  epoch: {epoch}  |  weight: {weight_label}')
    return model


def analyze(model, train_loader):
    all_ranks       = []   # (spatial,) per batch
    all_boundary_sv = []   # boundary SV per spatial position per batch
    rep_sv          = None

    with torch.no_grad():
        for X_batch, _, _ in train_loader:
            _, p  = model(X_batch.to(device))  # p: (spatial, 128, N_B)
            p     = p.detach().cpu()

            sv = torch.linalg.svdvals(p)      # (spatial, min(128, N_B))
            sv_np = sv.numpy()

            # energy-based rank per spatial slice
            energy       = sv ** 2
            cum_energy   = energy.cumsum(dim=1)
            total_energy = energy.sum(dim=1, keepdim=True)
            ranks        = (cum_energy / total_energy < eta).sum(dim=1) + 1  # (spatial,)
            ranks_np     = ranks.numpy()
            all_ranks.append(ranks_np)

            # boundary SV: the last singular value included in the effective rank
            boundary = sv_np[np.arange(len(ranks_np)), ranks_np - 1]  # (spatial,)
            all_boundary_sv.append(boundary)

            # save SV stats from first batch as representative
            if rep_sv is None:
                rep_sv = {
                    'mean': sv.mean(dim=0),
                    'q25':  torch.quantile(sv, 0.25, dim=0),
                    'q75':  torch.quantile(sv, 0.75, dim=0),
                    'min':  sv.min(dim=0).values,
                    'max':  sv.max(dim=0).values,
                    'K':    sv.shape[1],
                    'spatial': sv.shape[0],
                }

    ranks_per_batch    = np.stack(all_ranks)           # (n_batches, spatial)
    boundary_per_batch = np.stack(all_boundary_sv)     # (n_batches, spatial)
    return {
        'rep_sv':        rep_sv,
        'avg_ranks':     ranks_per_batch.mean(axis=0),     # (spatial,)
        'avg_boundary':  boundary_per_batch.mean(axis=0),  # (spatial,)
        'n_batches':     len(all_ranks),
    }


## ── Build dataloader (original training images, no shuffle) ──────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set    = MVTEC(root=root_path, train=True, transform=transform,
                     resize=im_shape, category=category)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
print(f'Training set size: {len(train_set)}  |  batch size: {batch_size}')

## ── Load each variant and run the SVD analysis ────────────────────────────────
results = {}
for variant in variants:
    model = load_model(variant)
    res   = analyze(model, train_loader)
    results[variant] = res

    avg_rank      = float(res['avg_ranks'].mean())
    rmin, rmax    = int(res['avg_ranks'].min()), int(np.ceil(res['avg_ranks'].max()))
    quantile_pcts = range(10, 91, 10)
    quantiles     = np.percentile(res['avg_ranks'], list(quantile_pcts))
    quantile_str  = ', '.join(f'Q{p}={q:.1f}' for p, q in zip(quantile_pcts, quantiles))
    print(f'[{variant}] Spatial positions: {res["rep_sv"]["spatial"]}  |  batches averaged: {res["n_batches"]}')
    print(f'[{variant}] Energy-based rank (η={eta}): avg={avg_rank:.2f}, min={rmin}, max={rmax}')
    print(f'[{variant}] Rank quantiles: {quantile_str}')
    print(f'[{variant}] Boundary SV (last SV in effective rank):  '
          f'mean={res["avg_boundary"].mean():.4f},  '
          f'min={res["avg_boundary"].min():.4f},  '
          f'max={res["avg_boundary"].max():.4f}')

variant_tag = '_vs_'.join(variants)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

## ── 1×2 comparison figure: SV spectrum + effective-rank histogram ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: SV decay (mean ± IQR across spatial positions), one curve per variant
for i, variant in enumerate(variants):
    rep_sv = results[variant]['rep_sv']
    K      = rep_sv['K']
    x      = np.arange(1, K + 1)
    color  = colors[i % len(colors)]
    label  = variant_labels.get(variant, variant)
    axes[0].plot(x, rep_sv['mean'], color=color, label=label)
    axes[0].fill_between(x, rep_sv['q25'], rep_sv['q75'], color=color, alpha=0.2,
                         label=f'{label}  25%-75%')
axes[0].set_xlabel('Singular Value Index', fontsize=fontsize_label)
axes[0].set_ylabel('Singular Value', fontsize=fontsize_label)
axes[0].set_title('Singular Value Spectrum', fontsize=fontsize_title)
axes[0].tick_params(labelsize=fontsize_tick)
axes[0].legend(fontsize=fontsize_legend, loc='upper right')
axes[0].grid(True)

# Right: histogram of effective rank across all batches × all spatial positions,
# on shared bins so the variants are visually comparable
all_avg_ranks = np.concatenate([results[v]['avg_ranks'] for v in variants])
rmin, rmax    = int(all_avg_ranks.min()), int(np.ceil(all_avg_ranks.max()))
bins          = range(rmin, rmax + 2)
for i, variant in enumerate(variants):
    color    = colors[i % len(colors)]
    label    = variant_labels.get(variant, variant)
    axes[1].hist(results[variant]['avg_ranks'], bins=bins, color=color, alpha=0.5,
                 edgecolor='black', align='left', label=label)
    q50 = float(np.percentile(results[variant]['avg_ranks'], 50))
    axes[1].axvline(q50, linestyle=':', color=color, label=f'{label}  $\\mathrm{{Pct}}_{{50}}(r_{{\\mathrm{{eff}}}})={q50:.1f}$')
    q90 = float(np.percentile(results[variant]['avg_ranks'], 90))
    axes[1].axvline(q90, linestyle='--', color=color, label=f'{label}  $\\mathrm{{Pct}}_{{90}}(r_{{\\mathrm{{eff}}}})={q90:.1f}$')
axes[1].set_xlabel('Effective Rank', fontsize=fontsize_label)
axes[1].set_ylabel('Frequency', fontsize=fontsize_label)
axes[1].set_title('Histogram of Batch-Averaged Effective Rank', fontsize=fontsize_title)
axes[1].set_xticks(range((rmin // 5) * 5, rmax + 5, 5))
axes[1].tick_params(labelsize=fontsize_tick)
axes[1].legend(fontsize=fontsize_legend, loc='upper right')

tnr = {'fontname': 'Times New Roman', 'fontsize': fontsize_label + 6}
for ax, letter in zip(axes, ['(a)', '(b)']):
    ax.text(0.5, -0.25, letter, transform=ax.transAxes,
            va='top', ha='center', **tnr)

plt.tight_layout()
plt.savefig(f'temp_var\\rank_compare_{category}_{variant_tag}_{weight_label}.png', dpi=150, bbox_inches='tight')
plt.show()

## ── Heatmaps: average rank + boundary SV per variant (shared color scale) ────
spatial   = results[variants[0]]['rep_sv']['spatial']
grid_size = int(spatial ** 0.5)

rank_all = np.stack([results[v]['avg_ranks'] for v in variants])
bsv_all  = np.stack([results[v]['avg_boundary'] for v in variants])
rank_vmin, rank_vmax = rank_all.min(), rank_all.max()
bsv_vmin, bsv_vmax   = bsv_all.min(), bsv_all.max()

fig_hm, axes_hm = plt.subplots(2, len(variants), figsize=(5 * len(variants), 8), squeeze=False)

for i, variant in enumerate(variants):
    label     = variant_labels.get(variant, variant)
    grid_rank = results[variant]['avg_ranks'].reshape(grid_size, grid_size)
    grid_bsv  = results[variant]['avg_boundary'].reshape(grid_size, grid_size)

    hm0 = axes_hm[0, i].imshow(grid_rank, cmap='viridis', vmin=rank_vmin, vmax=rank_vmax)
    plt.colorbar(hm0, ax=axes_hm[0, i], label='Avg Effective Rank')
    axes_hm[0, i].set_title(f'{label}' + r' — Avg Effective Rank of $P_{(m,n)}$')
    axes_hm[0, i].set_xlabel('Spatial column')
    axes_hm[0, i].set_ylabel('Spatial row')

    hm1 = axes_hm[1, i].imshow(grid_bsv, cmap='viridis', vmin=bsv_vmin, vmax=bsv_vmax)
    plt.colorbar(hm1, ax=axes_hm[1, i], label='Boundary SV')
    axes_hm[1, i].set_title(f'{label}' + r' — Boundary SV of $P_{(m,n)}$')
    axes_hm[1, i].set_xlabel('Spatial column')
    axes_hm[1, i].set_ylabel('Spatial row')

plt.suptitle(f'{category}  ({weight_label})', y=1.02)
plt.tight_layout()
plt.savefig(f'temp_var\\latent_rank_visualization_compare_{category}_{variant_tag}_{weight_label}.png', dpi=150, bbox_inches='tight')
plt.show()
