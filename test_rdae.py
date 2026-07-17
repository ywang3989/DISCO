import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision
import sklearn.metrics
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utlis import MVTEC, dice_coefficient, DMS_f, peak_signal_to_noise_ratio, structural_similarity, plot_result

## ── Config ────────────────────────────────────────────────────────────────────
category       = 'BTAD3_sys'
im_shape       = 128
strides        = (2, 3, 4)
training_epoch = 205

whether_plot  = True

# Pre-stored best thresholds per category × model_tag (fill in after running with whether_plot=False)
thre_best_map = {
    '02_sim1':                  {'rdae_s122': 0.14, 'rdae_s123': 0.17, 'rdae_s222': 0.0, 'rdae_s234': 0.29, 'rdae_s334': 0.42},
    '02_sim2':                  {'rdae_s122': 0.19, 'rdae_s123': 0.14, 'rdae_s222': 0.0, 'rdae_s234': 0.27, 'rdae_s334': 0.48},
    '02_sim3':                  {'rdae_s122': 0.15, 'rdae_s123': 0.23, 'rdae_s222': 0.0, 'rdae_s234': 0.21, 'rdae_s334': 0.35},
    'BTAD3_sys':                {'rdae_s122': 0.0, 'rdae_s123': 0.0, 'rdae_s222': 0.0, 'rdae_s234': 0.43, 'rdae_s334': 0.0},
    '02_sim_contam_level_0.05': {'rdae_s122': 0.0, 'rdae_s123': 0.0, 'rdae_s222': 0.0, 'rdae_s234': 0.30, 'rdae_s334': 0.0},
    '02_sim_contam_level_0.08': {'rdae_s122': 0.0, 'rdae_s123': 0.0, 'rdae_s222': 0.0, 'rdae_s234': 0.27, 'rdae_s334': 0.0},
    '02_sim_contam_level_0.1':  {'rdae_s122': 0.0, 'rdae_s123': 0.0, 'rdae_s222': 0.0, 'rdae_s234': 0.28, 'rdae_s334': 0.0},
}

## ── Auto-derived ──────────────────────────────────────────────────────────────
kernel_size   = 5
channel_input = 3
# btad_sim has defect_background (L ground truth); BTAD3_sys does not
dataset       = 'btad_sim' if (category.startswith('02_sim')) else 'btad'
thre_upper    = 2.0

# Defect type grouping for categorical Dice
if category == '02_sim1':
    defect_groups = {'blob': [0, 2, 4, 6], 'line': [1, 3, 5, 7]}
elif category == '02_sim2':
    defect_groups = {'blob': [0, 2, 3, 4], 'line': [1, 5, 6, 7]}
elif category == '02_sim3':
    defect_groups = {'blob': [1, 3, 6, 7], 'line': [0, 2, 4, 5]}
elif category == 'BTAD3_sys':
    defect_groups = {
        'crack':     list(range(900, 904)),
        'irregular': list(range(904, 908)),
        'large':     list(range(908, 912)),
        'scratch':   list(range(912, 916)),
        'spot':      list(range(916, 920)),
    }
else:
    defect_groups = {}
idx_to_type = {idx: t for t, idxs in defect_groups.items() for idx in idxs}

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

stride_tag = ''.join(map(str, strides))
model_tag  = f'rdae_s{stride_tag}'
batch_size = batch_size_map[category]
root_path  = 'data\\' + data_path_map[category]
out_dir    = f'temp_var\\rdae_bottleneck_variants\\{category}'

S_gtruth_path = root_path + '\\' + category + '\\train\\defect_ground_truth\\'
L_gtruth_path = root_path + '\\' + category + '\\train\\defect_background\\'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}  |  category: {category}  |  model: {model_tag}  |  epoch: {training_epoch}')

## ── Model ─────────────────────────────────────────────────────────────────────
class LRAE_2d_custom(nn.Module):
    def __init__(self, chnum_in, kernel_size, strides, im_shape):
        super().__init__()
        k  = kernel_size
        pd = 1
        s1, s2, s3 = strides
        f1, f2, f3 = 16, 64, 128

        enc1 = nn.Sequential(nn.Conv2d(chnum_in, f1, k, stride=s1, padding=pd),
                             nn.BatchNorm2d(f1), nn.LeakyReLU(0.2, inplace=True))
        enc2 = nn.Sequential(nn.Conv2d(f1, f2, k, stride=s2, padding=pd),
                             nn.BatchNorm2d(f2), nn.LeakyReLU(0.2, inplace=True))
        enc3 = nn.Sequential(nn.Conv2d(f2, f3, k, stride=s3, padding=pd),
                             nn.BatchNorm2d(f3), nn.LeakyReLU(0.2, inplace=True))
        self.encoder = nn.Sequential(enc1, enc2, enc3)

        with torch.no_grad():
            d  = torch.zeros(1, chnum_in, im_shape, im_shape)
            h1 = enc1(d);  H0 = im_shape
            h2 = enc2(h1); H1 = h1.shape[2]
            h3 = enc3(h2); H2 = h2.shape[2]; H3 = h3.shape[2]

        def out_pad(H_target, H_in, s):
            return H_target - ((H_in - 1) * s - 2 * pd + k)

        op1 = out_pad(H2, H3, s3)
        op2 = out_pad(H1, H2, s2)
        op3 = out_pad(H0, H1, s1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f3, f2, k, stride=s3, padding=pd, output_padding=op1),
            nn.BatchNorm2d(f2), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(f2, f1, k, stride=s2, padding=pd, output_padding=op2),
            nn.BatchNorm2d(f1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(f1, chnum_in, k, stride=s1, padding=pd, output_padding=op3),
        )

    def forward(self, x):
        z   = self.encoder(x)
        s   = z.data.shape
        pos = z.permute(2, 3, 1, 0).contiguous().view(-1, s[1], s[0])
        out = self.decoder(z)
        return out, pos


## ── Load model + saved decomposition ─────────────────────────────────────────
model = LRAE_2d_custom(chnum_in=channel_input, kernel_size=kernel_size,
                       strides=strides, im_shape=im_shape).to(device)
weight_path = (f'{out_dir}\\weight_{category}_{model_tag}'
               f'_{kernel_size}_{training_epoch}.pth')
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()
print(f'Loaded weights: {weight_path}')

L_train = torch.load(f'{out_dir}\\L_{category}_{model_tag}_{kernel_size}_{training_epoch}.pt',
                     map_location='cpu')
S_train = torch.load(f'{out_dir}\\S_{category}_{model_tag}_{kernel_size}_{training_epoch}.pt',
                     map_location='cpu')
print(f'L_train: {L_train.shape}  |  S_train: {S_train.shape}')

## ── Data loading (defective training samples) ─────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set_defect    = MVTEC(root=root_path, train=True, transform=transform,
                            resize=im_shape, category=category, train_defect=True)
train_loader_defect = DataLoader(train_set_defect, batch_size=batch_size, shuffle=False)
resizeTransf        = transforms.Resize(im_shape, 2, antialias=True)
print(f'Defective training set size: {len(train_set_defect)}')

## ── Threshold sweep (skipped when whether_plot=True) ─────────────────────────
if whether_plot:
    best_thre = thre_best_map[category][model_tag]
    print(f'Visualization mode — using stored threshold {best_thre:.4f}')

if not whether_plot:
    thresholds   = np.linspace(0.01, thre_upper, int(100 * thre_upper))
    metrics_thre = np.zeros((len(thresholds), 8))

    for idx, thre in enumerate(thresholds):
        dice_coefs = []
        auroc_vals = []
        bkgd_psnrs = []
        bkgd_ssims = []

        for _, (X_defect, _, indices) in enumerate(train_loader_defect):
            for i in range(X_defect.shape[0]):
                S_defect_show = np.sum(
                    np.abs(S_train[indices[i]].permute(1, 2, 0).cpu().detach().numpy()),
                    axis=2)
                S_mask_pred = (S_defect_show > thre).astype(int)

                S_ground_truth = torchvision.io.read_image(
                    S_gtruth_path + str(indices[i].item()) + '.png')
                S_ground_truth = resizeTransf(S_ground_truth).squeeze().detach().numpy()
                S_ground_truth[S_ground_truth != 0] = 1
                if S_ground_truth.ndim == 3 and S_ground_truth.shape[0] == 4:
                    S_ground_truth = S_ground_truth[0, :, :]

                dice_coefs.append(dice_coefficient(S_ground_truth, S_mask_pred))
                auroc_vals.append(
                    sklearn.metrics.roc_auc_score(S_ground_truth.flatten(), S_mask_pred.flatten()))

                if dataset == 'btad_sim':
                    L_backgd = L_train[indices[i]].view(1, channel_input, im_shape, im_shape)
                    L_backgd_ = ((L_backgd * 127.5 + 127.5)
                                 .squeeze().permute(1, 2, 0).numpy().astype(np.int16))
                    L_gt_show = mpimg.imread(L_gtruth_path + str(indices[i].item()) + '.png')
                    if L_gt_show.shape[2] == 4:
                        L_gt_show = L_gt_show[:, :, :3]
                    L_gt = transform(resizeTransf(
                        Image.fromarray((L_gt_show * 255).astype(np.uint8))))
                    L_gt_ = ((L_gt * 127.5 + 127.5)
                             .squeeze().permute(1, 2, 0).numpy().astype(np.int16))
                    K = 20
                    interpol_pts = np.linspace(0, 1, num=K + 1)
                    _, dms_psnr = DMS_f(S_ground_truth, interpol_pts,
                                        peak_signal_to_noise_ratio, L_gt_, L_backgd_)
                    _, dms_ssim = DMS_f(S_ground_truth, interpol_pts,
                                        structural_similarity, L_gt_, L_backgd_)
                else:
                    dms_psnr, dms_ssim = 0., 0.

                bkgd_psnrs.append(dms_psnr)
                bkgd_ssims.append(dms_ssim)

        metrics_thre[idx, 0] = np.mean(dice_coefs)
        metrics_thre[idx, 1] = np.std(dice_coefs,  ddof=1)
        metrics_thre[idx, 2] = np.mean(auroc_vals)
        metrics_thre[idx, 3] = np.std(auroc_vals,  ddof=1)
        metrics_thre[idx, 4] = np.mean(bkgd_psnrs)
        metrics_thre[idx, 5] = np.std(bkgd_psnrs,  ddof=1)
        metrics_thre[idx, 6] = np.mean(bkgd_ssims)
        metrics_thre[idx, 7] = np.std(bkgd_ssims,  ddof=1)

        if (idx + 1) % 20 == 0:
            print(f'  [{idx+1}/{len(thresholds)}]  thre={thre:.3f}  '
                  f'Dice={metrics_thre[idx,0]:.4f}  AUROC={metrics_thre[idx,2]:.4f}')

    ## ── Results at best Dice threshold ───────────────────────────────────────
    max_idx   = np.argmax(metrics_thre[:, 0])
    best_thre = thresholds[max_idx]
    print('\n' + '=' * 70)
    print(f'{model_tag} on {category}  (epoch {training_epoch})')
    print(f'Best threshold: {best_thre:.4f}  (Dice maximised)')
    print(f'  Dice:     avg={metrics_thre[max_idx,0]:.4f}  std={metrics_thre[max_idx,1]:.4f}')
    print(f'  AUROC:    avg={metrics_thre[max_idx,2]:.4f}  std={metrics_thre[max_idx,3]:.4f}')
    print(f'  DMS-PSNR: avg={metrics_thre[max_idx,4]:.4f}  std={metrics_thre[max_idx,5]:.4f}')
    print(f'  DMS-SSIM: avg={metrics_thre[max_idx,6]:.4f}  std={metrics_thre[max_idx,7]:.4f}')
    print('=' * 70)

    ## ── Threshold vs metrics plot ─────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, metrics_thre,
             label=[f'Dice Avg: {metrics_thre[max_idx,0]:.3f}',
                    f'Dice Std: {metrics_thre[max_idx,1]:.3f}',
                    f'AUROC Avg: {metrics_thre[max_idx,2]:.3f}',
                    f'AUROC Std: {metrics_thre[max_idx,3]:.3f}',
                    rf'$\mathrm{{DMS}}_{{20}}$-PSNR Avg: {metrics_thre[max_idx,4]:.3f}',
                    rf'$\mathrm{{DMS}}_{{20}}$-PSNR Std: {metrics_thre[max_idx,5]:.3f}',
                    rf'$\mathrm{{DMS}}_{{20}}$-SSIM Avg: {metrics_thre[max_idx,6]:.3f}',
                    rf'$\mathrm{{DMS}}_{{20}}$-SSIM Std: {metrics_thre[max_idx,7]:.3f}'])
    for col in range(metrics_thre.shape[1]):
        plt.plot(best_thre, metrics_thre[max_idx, col], marker='x', color='cyan')
    plt.axvline(best_thre, linestyle='--', color='gray', linewidth=0.8)
    plt.text(best_thre + 0.02, metrics_thre[max_idx, 0],
             f'Thre: {best_thre:.3f}', fontsize=9)
    plt.xlabel('Threshold')
    plt.title(f'Epoch {training_epoch} — {model_tag} on {category}')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}\\thre_curve_{category}_{model_tag}_{training_epoch}.png', dpi=150)
    plt.show()

## ── Per-sample results at best threshold ──────────────────────────────────────
print(f'\nCollecting per-sample results at threshold {best_thre:.4f} ...')
rows = []
# sim: grouping is by iteration position (matches test_RobMemeAE.py)
# BTAD3_sys: grouping is by file index (900-919)
use_pos_grouping = category.startswith('02_sim')
sample_pos = 0
for _, (X_defect, _, indices) in enumerate(train_loader_defect):
    for i in range(X_defect.shape[0]):
        S_defect_show = np.sum(
            np.abs(S_train[indices[i]].permute(1, 2, 0).cpu().detach().numpy()), axis=2)
        S_mask_pred = (S_defect_show > best_thre).astype(int)

        S_ground_truth = torchvision.io.read_image(
            S_gtruth_path + str(indices[i].item()) + '.png')
        S_ground_truth = resizeTransf(S_ground_truth).squeeze().detach().numpy()
        S_ground_truth[S_ground_truth != 0] = 1
        if S_ground_truth.ndim == 3 and S_ground_truth.shape[0] == 4:
            S_ground_truth = S_ground_truth[0, :, :]

        dice = dice_coefficient(S_ground_truth, S_mask_pred)

        if dataset == 'btad_sim':
            L_backgd = L_train[indices[i]].view(1, channel_input, im_shape, im_shape)
            L_backgd_ = ((L_backgd * 127.5 + 127.5)
                         .squeeze().permute(1, 2, 0).numpy().astype(np.int16))
            L_gt_show = mpimg.imread(L_gtruth_path + str(indices[i].item()) + '.png')
            if L_gt_show.shape[2] == 4:
                L_gt_show = L_gt_show[:, :, :3]
            L_gt = transform(resizeTransf(
                Image.fromarray((L_gt_show * 255).astype(np.uint8))))
            L_gt_ = ((L_gt * 127.5 + 127.5)
                     .squeeze().permute(1, 2, 0).numpy().astype(np.int16))
            K = 20
            interpol_pts = np.linspace(0, 1, num=K + 1)
            _, dms_psnr = DMS_f(S_ground_truth, interpol_pts,
                                peak_signal_to_noise_ratio, L_gt_, L_backgd_)
            _, dms_ssim = DMS_f(S_ground_truth, interpol_pts,
                                structural_similarity, L_gt_, L_backgd_)
        else:
            dms_psnr, dms_ssim = 0., 0.

        auroc_val = sklearn.metrics.roc_auc_score(
            S_ground_truth.flatten(), S_defect_show.flatten())

        if whether_plot:
            X_image_show  = X_defect[i].permute(1, 2, 0).cpu().numpy()
            L_backgd_show = np.clip(
                L_train[indices[i]].permute(1, 2, 0).numpy(), -1., 1.)
            if dataset == 'btad_sim':
                L_gt_rgb = L_gt_show
            else:
                L_gt_rgb = np.zeros((im_shape, im_shape, 3))
            E_noise_show = np.zeros((im_shape, im_shape, 1))
            plot_result(X_image_show, L_backgd_show, L_gt_rgb,
                        S_defect_show, E_noise_show, S_mask_pred, S_ground_truth,
                        [dice, auroc_val, dms_psnr, dms_ssim])

        row = {'category': category, 'sample_idx': indices[i].item(),
               'dice': dice, 'dms_psnr': dms_psnr, 'dms_ssim': dms_ssim}
        if defect_groups:
            group_key = sample_pos if use_pos_grouping else indices[i].item()
            row['defect_type'] = idx_to_type.get(group_key, 'unknown')
        rows.append(row)
        sample_pos += 1

df_new = pd.DataFrame(rows)

# ── Build summary ──────────────────────────────────────────────────────────────
def _grp_stats(df, mask=None):
    sub = df if mask is None else df[mask]
    return {
        'n':             len(sub),
        'dice_mean':     sub['dice'].mean(),
        'dice_std':      sub['dice'].std(ddof=1) if len(sub) > 1 else 0.,
        'dms_psnr_mean': sub['dms_psnr'].mean(),
        'dms_psnr_std':  sub['dms_psnr'].std(ddof=1) if len(sub) > 1 else 0.,
        'dms_ssim_mean': sub['dms_ssim'].mean(),
        'dms_ssim_std':  sub['dms_ssim'].std(ddof=1) if len(sub) > 1 else 0.,
    }

summary_rows = [{'category': category, 'defect_type': 'overall', **_grp_stats(df_new)}]
if defect_groups:
    for type_name in defect_groups:
        summary_rows.append({
            'category': category, 'defect_type': type_name,
            **_grp_stats(df_new, df_new['defect_type'] == type_name)
        })
df_summary = pd.DataFrame(summary_rows)

# Print results
print(f'\nResults at threshold {best_thre:.4f}:')
for _, row in df_summary.iterrows():
    print(f'  {row["defect_type"]:12s}:  '
          f'Dice={row["dice_mean"]:.4f}±{row["dice_std"]:.4f}  '
          f'PSNR={row["dms_psnr_mean"]:.4f}±{row["dms_psnr_std"]:.4f}  '
          f'SSIM={row["dms_ssim_mean"]:.4f}±{row["dms_ssim_std"]:.4f}  '
          f'(n={int(row["n"])})')

# Save per-category CSVs (individual samples + summary)
cat_csv = f'{out_dir}\\individual_{category}_{model_tag}_{training_epoch}.csv'
df_new.to_csv(cat_csv, index=False)
print(f'Saved: {cat_csv}')

sum_csv = f'{out_dir}\\summary_{category}_{model_tag}_{training_epoch}.csv'
df_summary.to_csv(sum_csv, index=False)
print(f'Saved: {sum_csv}')

# Update combined CSVs for sim1/2/3
if category in ['02_sim1', '02_sim2', '02_sim3']:
    base = f'temp_var\\rdae_bottleneck_variants'

    # Combined individual
    combined_csv = f'{base}\\{model_tag}_sim_combined.csv'
    if os.path.exists(combined_csv):
        df_ex = pd.read_csv(combined_csv)
        df_ex = df_ex[df_ex['category'] != category]
        df_combined = pd.concat([df_ex, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(combined_csv, index=False)

    # Combined summary
    combined_sum_csv = f'{base}\\{model_tag}_sim_combined_summary.csv'
    if os.path.exists(combined_sum_csv):
        df_sum_ex = pd.read_csv(combined_sum_csv)
        df_sum_ex = df_sum_ex[df_sum_ex['category'] != category]
        df_sum_combined = pd.concat([df_sum_ex, df_summary], ignore_index=True)
    else:
        df_sum_combined = df_summary
    df_sum_combined.to_csv(combined_sum_csv, index=False)

    present = sorted(df_combined['category'].unique())
    print(f'\nUpdated combined CSVs ({present}, {len(df_combined)} samples):')
    print(f'  {combined_csv}')
    print(f'  {combined_sum_csv}')
    # Print combined categorical breakdown
    print(f'  Combined categorical Dice:')
    for type_name in ['overall'] + list(defect_groups.keys()):
        if type_name == 'overall':
            sub = df_combined
        else:
            sub = df_combined[df_combined['defect_type'] == type_name]
        if len(sub) == 0:
            continue
        print(f'    {type_name:12s}:  mean={sub["dice"].mean():.4f}  '
              f'std={sub["dice"].std(ddof=1):.4f}  n={len(sub)}')
