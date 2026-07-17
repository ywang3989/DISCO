import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.size'] = 14

## Data
anomaly_ratios = [2, 5, 8, 10]
x = np.array(anomaly_ratios)

models = ['rpca', 'rdae', 'memae', 'draem', 'disco']
display_name = {'rpca': 'RPCA', 'rdae': 'RDAE', 'memae': 'MemAE', 'draem': 'DRAEM', 'disco': 'DISCO (Ours)'}
colors  = {'rpca': 'C0',      'rdae': 'C1',         'memae': 'C2',         'draem': 'C3',     'disco': 'C4'}
markers = {'rpca': 'o',       'rdae': 's',          'memae': '^',          'draem': 'D',      'disco': '*'}
lws     = {'rpca': 1.5,       'rdae': 1.5,          'memae': 1.5,          'draem': 1.5,      'disco': 2.5}
mss     = {'rpca': 7,         'rdae': 7,            'memae': 8,            'draem': 7,        'disco': 10}

## --- Dice (mean ± std) ---
dice_mean = {
    'rpca':  [0.130, 0.140, 0.168, 0.172],
    'rdae':  [0.453, 0.502, 0.471, 0.460],
    'memae': [0.612, 0.573, 0.537, 0.517],
    'draem': [0.596, 0.501, 0.359, 0.306],
    'disco': [0.758, 0.685, 0.690, 0.693],
}
dice_std = {
    'rpca':  [0.172, 0.178, 0.237, 0.261],
    'rdae':  [0.217, 0.197, 0.225, 0.198],
    'memae': [0.088, 0.064, 0.096, 0.116],
    'draem': [0.150, 0.131, 0.198, 0.168],
    'disco': [0.145, 0.226, 0.166, 0.185],
}

## --- AUROC (pooled, no std) ---
auroc = {
    'rpca':  [0.5491, 0.5420, 0.5394, 0.5393],
    'rdae':  [0.9363, 0.9312, 0.9089, 0.9260],
    'memae': [0.9813, 0.9734, 0.9597, 0.9616],
    'draem': [0.9736, 0.9175, 0.7010, 0.7171],
    'disco': [0.9879, 0.9754, 0.9812, 0.9837],
}

## --- AUPRC (pooled, no std) ---
auprc = {
    'rpca':  [0.5656, 0.5602, 0.5612, 0.5580],
    'rdae':  [0.3978, 0.4386, 0.3812, 0.3698],
    'memae': [0.5593, 0.4923, 0.4480, 0.4329],
    'draem': [0.4962, 0.3748, 0.1751, 0.1797],
    'disco': [0.8308, 0.7812, 0.7850, 0.7970],
}

## --- DMS20-PSNR (mean ± std) ---
psnr_mean = {
    'rpca':  [28.971, 30.715, 30.002, 29.779],
    'rdae':  [28.918, 29.889, 29.671, 29.343],
    'memae': [28.911, 30.693, 29.874, 29.701],
    'draem': [27.746, 29.464, 28.900, 28.909],
    'disco': [31.150, 32.056, 31.708, 31.392],
}
psnr_std = {
    'rpca':  [3.191, 4.165, 3.767, 3.563],
    'rdae':  [2.689, 2.248, 2.693, 2.387],
    'memae': [3.462, 3.962, 3.632, 3.418],
    'draem': [3.138, 4.094, 3.484, 3.395],
    'disco': [3.230, 2.763, 3.242, 2.865],
}

## --- DMS20-SSIM (mean ± std) ---
ssim_mean = {
    'rpca':  [0.947, 0.961, 0.956, 0.952],
    'rdae':  [0.953, 0.965, 0.960, 0.959],
    'memae': [0.944, 0.961, 0.955, 0.952],
    'draem': [0.932, 0.950, 0.947, 0.944],
    'disco': [0.967, 0.976, 0.973, 0.971],
}
ssim_std = {
    'rpca':  [0.037, 0.035, 0.038, 0.040],
    'rdae':  [0.030, 0.023, 0.031, 0.028],
    'memae': [0.040, 0.036, 0.039, 0.041],
    'draem': [0.044, 0.041, 0.041, 0.043],
    'disco': [0.026, 0.021, 0.025, 0.026],
}

xtick_labels = ['2%', '5%', '8%', '10%']


def plot_metric_no_std(ax, mean_data, std_data, ylabel, ylim=None):
    for m in models:
        mu = np.array(mean_data[m])
        sd = np.array(std_data[m])
        ax.errorbar(x, mu, yerr=sd, label=display_name[m],
                    color=colors[m], marker=markers[m], linewidth=lws[m],
                    markersize=mss[m], capsize=4)
    ax.set_xlabel('Anomaly Ratio')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)


def plot_metric_no_std(ax, data, ylabel, ylim=None):
    for m in models:
        ax.plot(x, data[m], label=display_name[m],
                color=colors[m], marker=markers[m], linewidth=lws[m],
                markersize=mss[m])
    ax.set_xlabel('Anomaly Ratio')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)


## ── Combined 2×3 figure ─────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 0
plot_metric_no_std(axes[0, 0], dice_mean,  r'Dice',                        ylim=(0,    1.05))
plot_metric_no_std(axes[0, 1], auroc,       r'AUROC',                       ylim=(0.4,  1.05))
plot_metric_no_std(axes[0, 2], auprc,       r'AUPRC',                       ylim=(0.1,  1.0))

# Row 1: first cell empty, then PSNR and SSIM
axes[1, 0].set_visible(False)
plot_metric_no_std(axes[1, 1], psnr_mean,  r'$\mathrm{DMS}_{20}$-PSNR',   ylim=(20,   40))
plot_metric_no_std(axes[1, 2], ssim_mean,  r'$\mathrm{DMS}_{20}$-SSIM',   ylim=(0.85, 1.02))

# Remove per-panel legends
active_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]
for ax in active_axes:
    ax.get_legend().remove()

# Add (a)–(e) labels in Times New Roman, below x-axis label
tnr = {'fontname': 'Times New Roman', 'fontsize': 24}
for ax, letter in zip(active_axes, ['(a)', '(b)', '(c)', '(d)', '(e)']):
    ax.text(0.5, -0.17, letter, transform=ax.transAxes,
            va='top', ha='center', **tnr)

# Shared legend at top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=13,
           bbox_to_anchor=(0.5, 1.03), frameon=True)

plt.tight_layout()
plt.savefig('temp_var\\sensitivity_combined.png', dpi=150, bbox_inches='tight')
plt.show()
