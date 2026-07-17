from utlis import *
from models import *
from torch.utils.data import DataLoader
import sklearn.metrics
import torchvision
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = ('cuda' if torch.cuda.is_available() else 'cpu')

log_file = open('temp_var\\roc_pixelevel_results.txt', 'w')

def log(msg=''):
    print(msg)
    log_file.write(str(msg) + '\n')

log(f'Using {device} device')


## ── Mode matrix  (C = eval_contam_levels,  V = eval_rdae_variants) ─────────────
#
#   C=False  V=False  │  Other models   on sim1/2/3 + BTAD3_sys   │  roc_prc_sim_combined.png
#   C=True   V=False  │  Other models   on contam 0.05/0.08/0.10  │  —
#   C=False  V=True   │  RDAE variants  on sim1/2/3 + BTAD3_sys   │  roc_prc_sim_combined_rdae_variants.png
#   C=True   V=True   │  RDAE variants  on contam 0.05/0.08/0.10  │  —
#
#   All modes: individual ROC+PRC per dataset (all evaluated models overlaid).
#   BTAD3_sys: individual plot only — excluded from the combined sim plot.
#
## ──────────────────────────────────────────────────────────────────────────────

eval_contam_levels = False
eval_rdae_variants = False



## Config
channel_input = 3
kernel_size = 5
batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model_para = {
    '02_sim1':   {'disco': [106, 0.14], 'disco-wo-p': [105, 0.06], 'disco-wo-e': [105, 0.23],
                  'disco-wo-ep': [108, 0.07], 'rdae': [103, 0.08], 'rpca': [0, 0.01], 'memae': [500, 0.14]},
    '02_sim2':   {'disco': [104, 0.17], 'disco-wo-p': [105, 0.07], 'disco-wo-e': [105, 0.25],
                  'disco-wo-ep': [108, 0.09], 'rdae': [103, 0.11], 'rpca': [0, 0.01], 'memae': [500, 0.13]},
    '02_sim3':   {'disco': [104, 0.16], 'disco-wo-p': [105, 0.04], 'disco-wo-e': [104, 0.22],
                  'disco-wo-ep': [107, 0.08], 'rdae': [103, 0.17], 'rpca': [0, 0.01], 'memae': [500, 0.13]},
    'BTAD3_sys': {'disco': [213, 0.21], 'disco-wo-p': [209, 0.15], 'disco-wo-e': [214, 0.43],
                  'disco-wo-ep': [212, 0.47], 'rdae': [203, 0.40], 'rpca': [0, 0.01], 'memae': [2000, 0.32]},
    '02_sim_contam_level_0.05': {'disco': [105, 0.16], 'rdae': [105, 0.26], 'rpca': [0, 0.01], 'memae': [500, 0.13]},
    '02_sim_contam_level_0.08': {'disco': [105, 0.16], 'rdae': [106, 0.24], 'rpca': [0, 0.01], 'memae': [500, 0.12]},
    '02_sim_contam_level_0.1':  {'disco': [104, 0.16], 'rdae': [105, 0.25], 'rpca': [0, 0.01], 'memae': [500, 0.12]}
}

# blob/line index splits for original btad_sim datasets
blob_line_index = {
    '02_sim1': {'blob': [0, 2, 4, 6], 'line': [1, 3, 5, 7]},
    '02_sim2': {'blob': [0, 2, 3, 4], 'line': [1, 5, 6, 7]},
    '02_sim3': {'blob': [1, 3, 6, 7], 'line': [0, 2, 4, 5]},
}

rdae_variants = [
    {'tag': 'rdae_s122', 'strides': (1, 2, 2)},  # 30×30
    {'tag': 'rdae_s123', 'strides': (1, 2, 3)},  # 20×20
    {'tag': 'rdae_s222', 'strides': (2, 2, 2)},  # 15×15
    {'tag': 'rdae_s234', 'strides': (2, 3, 4)},  # 5×5
    {'tag': 'rdae_s334', 'strides': (3, 3, 4)},  # 3×3
]

# Fill in the ADMM epoch actually saved for each variant × category (0 = not yet trained)
rdae_variant_para = {
    '02_sim1':    {'rdae_s122': 105, 'rdae_s123': 106, 'rdae_s222': 106, 'rdae_s234': 104, 'rdae_s334': 104},
    '02_sim2':    {'rdae_s122': 104, 'rdae_s123': 105, 'rdae_s222': 106, 'rdae_s234': 105, 'rdae_s334': 103},
    '02_sim3':    {'rdae_s122': 107, 'rdae_s123': 105, 'rdae_s222': 106, 'rdae_s234': 104, 'rdae_s334': 104},
    'BTAD3_sys':  {'rdae_s122': 0,   'rdae_s123': 0,   'rdae_s222': 0,   'rdae_s234': 205,   'rdae_s334': 0  },
    '02_sim_contam_level_0.05': {'rdae_s122': 0, 'rdae_s123': 0, 'rdae_s222': 0, 'rdae_s234': 105, 'rdae_s334': 0},
    '02_sim_contam_level_0.08': {'rdae_s122': 0, 'rdae_s123': 0, 'rdae_s222': 0, 'rdae_s234': 105, 'rdae_s334': 0},
    '02_sim_contam_level_0.1':  {'rdae_s122': 0, 'rdae_s123': 0, 'rdae_s222': 0, 'rdae_s234': 105, 'rdae_s334': 0},
}

if not eval_contam_levels:
    datasets_config = [
        {'dataset': 'btad_sim', 'category': '02_sim1',   'im_shape': 128, 'data_path': 'btad_simulation\\02_sim_contam_level_0.02'},
        {'dataset': 'btad_sim', 'category': '02_sim2',   'im_shape': 128, 'data_path': 'btad_simulation\\02_sim_contam_level_0.02'},
        {'dataset': 'btad_sim', 'category': '02_sim3',   'im_shape': 128, 'data_path': 'btad_simulation\\02_sim_contam_level_0.02'},
        {'dataset': 'btad',     'category': 'BTAD3_sys', 'im_shape': 128, 'data_path': 'BTech_Dataset_Transformed'},
    ]
else:
    datasets_config = [
        {'dataset': 'btad_sim', 'category': '02_sim_contam_level_0.05', 'im_shape': 128, 'data_path': 'btad_simulation'},
        {'dataset': 'btad_sim', 'category': '02_sim_contam_level_0.08', 'im_shape': 128, 'data_path': 'btad_simulation'},
        {'dataset': 'btad_sim', 'category': '02_sim_contam_level_0.1',  'im_shape': 128, 'data_path': 'btad_simulation'},
    ]

models_to_eval = ['rpca', 'rdae', 'memae', 'draem', 'disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep']
memory_size = 500
shrink_threshold = 0.0001


def load_model_and_components(category, model_running, im_shape):
    """Load stored L/S/E tensors and model weights. Returns (model, L_train, S_train, E_train)."""
    model = None

    if model_running == 'draem':
        S_train = torch.load('temp_var\\' + category + '\\S_' + category + '_draem.pt')
        return None, None, S_train, None

    epoch = model_para[category][model_running][0]

    if model_running == 'rpca':
        L_train = torch.load('temp_var\\' + category + '\\L_' + category + '_' + model_running + '.pt')
        S_train = torch.load('temp_var\\' + category + '\\S_' + category + '_' + model_running + '.pt')
        E_train = None
    elif model_running == 'memae':
        model = MemAE_2d(chnum_in=channel_input, kernel_size=kernel_size, mem_dim=memory_size,
                         shrink_thres=shrink_threshold, low_rank=128, low_rank_processing=False).to(device)
        pth = 'temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch) + '.pth'
        model.load_state_dict(torch.load(pth))
        model.eval()
        L_train, S_train, E_train = None, None, None
    else:
        model = LRAE_2d(chnum_in=channel_input, kernel_size=kernel_size).to(device)
        pth = 'temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch) + '.pth'
        model.load_state_dict(torch.load(pth))
        model.eval()
        L_train = torch.load('temp_var\\' + category + '\\L_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch) + '.pt')
        S_train = torch.load('temp_var\\' + category + '\\S_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch) + '.pt')
        E_train = torch.load('temp_var\\' + category + '\\E_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch) + '.pt')

    return model, L_train, S_train, E_train


def get_anomaly_score(model_running, model, S_train, im_shape, X_defect_i, idx_i):
    """Return continuous pixel-level anomaly score map (H, W) for one sample."""
    if model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae', 'rpca', 'draem']:
        score = np.sum(np.abs(S_train[idx_i].permute(1, 2, 0).cpu().detach().numpy()), axis=2)
    elif model_running == 'memae':
        X_input = X_defect_i.view(1, channel_input, im_shape, im_shape).to(device)
        with torch.no_grad():
            L_backgd, _ = model(X_input)
        S_defect = X_input - L_backgd
        score = np.sum(np.abs(S_defect.squeeze().permute(1, 2, 0).cpu().detach().numpy()), axis=2)
    return score


def load_ground_truth(gtruth_path, idx, resize_transf):
    """Load and binarize ground truth mask for one sample."""
    gt = torchvision.io.read_image(gtruth_path + str(idx) + '.png')
    gt = resize_transf(gt).squeeze().detach().numpy()
    gt[gt != 0] = 1
    if gt.ndim == 3 and gt.shape[0] == 4:
        gt = gt[0, :, :]
    return gt.astype(np.int32)


display_name = {'rpca': 'RPCA', 'rdae': 'RDAE', 'memae': 'MemAE', 'disco': 'DISCO', 'draem': 'DRAEM'}

## Accumulators for combined simulation result
sim_combined_scores = {m: [] for m in models_to_eval}
sim_combined_labels = {m: [] for m in models_to_eval}

## Main evaluation loop
log('=' * 70)
log(f'{"Model":<14}  {"Dataset":<12}  {"AUROC(per-img)":<18}  {"AUPRC(per-img)":<18}  {"AUROC(pooled)":<16}  {"AUPRC(pooled)"}')
log('=' * 70)

# Main loop runs only for other-model evaluation (not RDAE variants)
datasets_for_main = [] if eval_rdae_variants else datasets_config

for cfg in datasets_for_main:
    dataset_type = cfg['dataset']
    category     = cfg['category']
    im_shape     = cfg['im_shape']
    root_path    = 'data\\' + cfg['data_path']
    gtruth_path  = root_path + '\\' + category + '\\train\\defect_ground_truth\\'

    resize_transf = transforms.Resize(im_shape, 2, antialias=True)

    train_set_defect = MVTEC(root=root_path, train=True, transform=transform,
                             resize=im_shape, category=category, train_defect=True)
    train_loader_defect = DataLoader(train_set_defect, batch_size=batch_size, shuffle=False)

    # Collect curves for all models, then plot together
    curves = {}

    for model_running in models_to_eval:
        try:
            model, L_train, S_train, E_train = load_model_and_components(category, model_running, im_shape)
        except FileNotFoundError as e:
            log(f'  [SKIP] {model_running} on {category}: {e}')
            continue

        per_img_auroc = []
        per_img_auprc = []
        all_scores    = []
        all_labels    = []

        for _, (X_defect, _, indices) in enumerate(train_loader_defect):
            for i in range(X_defect.shape[0]):
                score = get_anomaly_score(model_running, model, S_train, im_shape, X_defect[i], indices[i])
                gt    = load_ground_truth(gtruth_path, indices[i].item(), resize_transf)

                score_flat = score.flatten()
                gt_flat    = gt.flatten()

                all_scores.append(score_flat)
                all_labels.append(gt_flat)

                if gt_flat.sum() > 0:
                    per_img_auroc.append(sklearn.metrics.roc_auc_score(gt_flat, score_flat))
                    per_img_auprc.append(sklearn.metrics.average_precision_score(gt_flat, score_flat))

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        fpr, tpr, _ = sklearn.metrics.roc_curve(all_labels, all_scores)
        pooled_auroc = sklearn.metrics.auc(fpr, tpr)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(all_labels, all_scores)
        pooled_auprc = sklearn.metrics.auc(recall, precision)

        curves[model_running] = {
            'fpr': fpr, 'tpr': tpr, 'auroc': pooled_auroc,
            'recall': recall, 'precision': precision, 'auprc': pooled_auprc,
            'baseline': all_labels.mean(),
            'per_img_auroc': per_img_auroc, 'per_img_auprc': per_img_auprc,
        }

        if dataset_type == 'btad_sim':
            sim_combined_scores[model_running].append(all_scores)
            sim_combined_labels[model_running].append(all_labels)

        auroc_mean = np.mean(per_img_auroc)
        auroc_std  = np.std(per_img_auroc, ddof=1)
        auprc_mean = np.mean(per_img_auprc)
        auprc_std  = np.std(per_img_auprc, ddof=1)

        log(f'{model_running:<14}  {category:<12}  '
            f'{auroc_mean:.3f}±{auroc_std:.3f}        '
            f'{auprc_mean:.3f}±{auprc_std:.3f}        '
            f'{pooled_auroc:.4f}          '
            f'{pooled_auprc:.4f}')

        # Group-level breakdown
        if dataset_type == 'btad_sim' and category in blob_line_index:
            blob_idx = blob_line_index[category]['blob']
            line_idx = blob_line_index[category]['line']
            auroc_arr = np.array(per_img_auroc)
            auprc_arr = np.array(per_img_auprc)
            log(f'  blob AUROC: {np.round(auroc_arr[blob_idx], 3)}, mean={np.round(np.mean(auroc_arr[blob_idx]), 3):.3f}')
            log(f'  line AUROC: {np.round(auroc_arr[line_idx], 3)}, mean={np.round(np.mean(auroc_arr[line_idx]), 3):.3f}')
            log(f'  blob AUPRC: {np.round(auprc_arr[blob_idx], 3)}, mean={np.round(np.mean(auprc_arr[blob_idx]), 3):.3f}')
            log(f'  line AUPRC: {np.round(auprc_arr[line_idx], 3)}, mean={np.round(np.mean(auprc_arr[line_idx]), 3):.3f}')

        if dataset_type == 'btad':
            auroc_groups = np.reshape(np.array(per_img_auroc), (5, 4))
            auprc_groups = np.reshape(np.array(per_img_auprc), (5, 4))
            log(f'  Group AUROC avg: {np.round(np.mean(auroc_groups, axis=1), 3)}')
            log(f'  Group AUPRC avg: {np.round(np.mean(auprc_groups, axis=1), 3)}')

    ## Plot all models together for this dataset
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        baseline = list(curves.values())[0]['baseline']

        for model_running, c in curves.items():
            name = display_name.get(model_running, model_running)
            axes[0].plot(c['fpr'], c['tpr'],
                         label=f'{name} - AUROC: {c["auroc"]:.4f}')
            axes[1].plot(c['recall'], c['precision'],
                         label=f'{name} - AUPRC: {c["auprc"]:.4f}')

        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random - AUROC: 0.5000')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'Pixel-wise ROC - {category}')
        axes[0].legend(loc='lower right')

        axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=0.8,
                        label=f'Random - AUPRC: {baseline:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'Pixel-wise PRC - {category}')
        axes[1].legend(loc='lower left')

        plt.tight_layout()
        plt.savefig('temp_var\\roc_prc_' + category + '.png', dpi=150)
        plt.show()

log('=' * 70)

## Combined plot for other models on original sim1/2/3 (only this combination)
if not eval_contam_levels and not eval_rdae_variants:
    log('\n--- Simulation Combined (02_sim1 + 02_sim2 + 02_sim3) ---')
    sim_curves = {}
    for model_running in models_to_eval:
        if not sim_combined_scores[model_running]:
            continue
        scores = np.concatenate(sim_combined_scores[model_running])
        labels = np.concatenate(sim_combined_labels[model_running])

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, scores)
        pooled_auroc = sklearn.metrics.auc(fpr, tpr)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, scores)
        pooled_auprc = sklearn.metrics.auc(recall, precision)

        sim_curves[model_running] = {
            'fpr': fpr, 'tpr': tpr, 'auroc': pooled_auroc,
            'recall': recall, 'precision': precision, 'auprc': pooled_auprc,
            'baseline': labels.mean(),
        }
        log(f'{model_running:<14}  sim_combined  AUROC(pooled): {pooled_auroc:.4f}  AUPRC(pooled): {pooled_auprc:.4f}')

    if sim_curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        baseline = list(sim_curves.values())[0]['baseline']
        for model_running, c in sim_curves.items():
            name = display_name.get(model_running, model_running)
            axes[0].plot(c['fpr'], c['tpr'], label=f'{name} - AUROC: {c["auroc"]:.4f}')
            axes[1].plot(c['recall'], c['precision'], label=f'{name} - AUPRC: {c["auprc"]:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random - AUROC: 0.5000')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Pixel-wise ROC - Simulation Overall')
        axes[0].legend(loc='lower right')
        axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=0.8,
                        label=f'Random - AUPRC: {baseline:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Pixel-wise PRC - Simulation Overall')
        axes[1].legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('temp_var\\roc_prc_sim_combined.png', dpi=150)
        plt.show()

## ── RDAE bottleneck variants ──────────────────────────────────────────────────
if eval_rdae_variants:
    # datasets_config already set to the right datasets (original or contam) based on eval_contam_levels
    rdae_cats = datasets_config

    if eval_contam_levels:
        combined_log_label = 'Contam Combined (0.05 + 0.08 + 0.10)'
        combined_save      = 'temp_var\\roc_prc_contam_combined_rdae_variants.png'
        combined_roc_title = 'Pixel-wise ROC - Contam Overall (RDAE variants)'
        combined_prc_title = 'Pixel-wise PRC - Contam Overall (RDAE variants)'
    else:
        combined_log_label = 'Simulation Combined (02_sim1 + 02_sim2 + 02_sim3)'
        combined_save      = 'temp_var\\roc_prc_sim_combined_rdae_variants.png'
        combined_roc_title = 'Pixel-wise ROC - Simulation Combined (RDAE variants)'
        combined_prc_title = 'Pixel-wise PRC - Simulation Combined (RDAE variants)'

    var_tags = [v['tag'] for v in rdae_variants]
    var_comb_scores = {t: [] for t in var_tags}
    var_comb_labels = {t: [] for t in var_tags}

    log('\n' + '=' * 70)
    log('RDAE Bottleneck Variants')
    log(f'{"Model":<14}  {"Dataset":<12}  {"AUROC(per-img)":<18}  {"AUPRC(per-img)":<18}  {"AUROC(pooled)":<16}  {"AUPRC(pooled)"}')
    log('=' * 70)

    for cfg in rdae_cats:
        category = cfg['category']
        im_shape = cfg['im_shape']
        root_path   = 'data\\' + cfg['data_path']
        gtruth_path = root_path + '\\' + category + '\\train\\defect_ground_truth\\'
        var_dir     = f'temp_var\\rdae_bottleneck_variants\\{category}'

        resize_transf = transforms.Resize(im_shape, 2, antialias=True)
        train_set_defect = MVTEC(root=root_path, train=True, transform=transform,
                                 resize=im_shape, category=category, train_defect=True)
        train_loader_defect = DataLoader(train_set_defect, batch_size=batch_size, shuffle=False)

        curves = {}

        for v in rdae_variants:
            tag   = v['tag']
            epoch = rdae_variant_para[category][tag]
            if epoch == 0:
                log(f'  [SKIP] {tag} on {category}: epoch not set in rdae_variant_para')
                continue

            s_path = f'{var_dir}\\S_{category}_{tag}_{kernel_size}_{epoch}.pt'
            try:
                S_train = torch.load(s_path, map_location='cpu')
            except FileNotFoundError as e:
                log(f'  [SKIP] {tag} on {category}: {e}')
                continue

            per_img_auroc = []
            per_img_auprc = []
            all_scores    = []
            all_labels    = []

            for _, (X_defect, _, indices) in enumerate(train_loader_defect):
                for i in range(X_defect.shape[0]):
                    score = np.sum(
                        np.abs(S_train[indices[i]].permute(1, 2, 0).cpu().detach().numpy()),
                        axis=2)
                    gt = load_ground_truth(gtruth_path, indices[i].item(), resize_transf)

                    score_flat = score.flatten()
                    gt_flat    = gt.flatten()
                    all_scores.append(score_flat)
                    all_labels.append(gt_flat)

                    if gt_flat.sum() > 0:
                        per_img_auroc.append(
                            sklearn.metrics.roc_auc_score(gt_flat, score_flat))
                        per_img_auprc.append(
                            sklearn.metrics.average_precision_score(gt_flat, score_flat))

            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)

            fpr, tpr, _          = sklearn.metrics.roc_curve(all_labels, all_scores)
            pooled_auroc         = sklearn.metrics.auc(fpr, tpr)
            precision, recall, _ = sklearn.metrics.precision_recall_curve(all_labels, all_scores)
            pooled_auprc         = sklearn.metrics.auc(recall, precision)

            curves[tag] = {
                'fpr': fpr, 'tpr': tpr, 'auroc': pooled_auroc,
                'recall': recall, 'precision': precision, 'auprc': pooled_auprc,
                'baseline': all_labels.mean(),
                'per_img_auroc': per_img_auroc, 'per_img_auprc': per_img_auprc,
            }
            if cfg['dataset'] == 'btad_sim':   # sim1/2/3 only, not BTAD3_sys
                var_comb_scores[tag].append(all_scores)
                var_comb_labels[tag].append(all_labels)

            auroc_mean = np.mean(per_img_auroc)
            auroc_std  = np.std(per_img_auroc, ddof=1)
            auprc_mean = np.mean(per_img_auprc)
            auprc_std  = np.std(per_img_auprc, ddof=1)
            log(f'{tag:<14}  {category:<12}  '
                f'{auroc_mean:.3f}±{auroc_std:.3f}        '
                f'{auprc_mean:.3f}±{auprc_std:.3f}        '
                f'{pooled_auroc:.4f}          {pooled_auprc:.4f}')

            if category in blob_line_index:
                blob_idx  = blob_line_index[category]['blob']
                line_idx  = blob_line_index[category]['line']
                auroc_arr = np.array(per_img_auroc)
                auprc_arr = np.array(per_img_auprc)
                log(f'  blob AUROC: {np.round(auroc_arr[blob_idx], 3)}'
                    f', mean={np.mean(auroc_arr[blob_idx]):.3f}')
                log(f'  line AUROC: {np.round(auroc_arr[line_idx], 3)}'
                    f', mean={np.mean(auroc_arr[line_idx]):.3f}')
                log(f'  blob AUPRC: {np.round(auprc_arr[blob_idx], 3)}'
                    f', mean={np.mean(auprc_arr[blob_idx]):.3f}')
                log(f'  line AUPRC: {np.round(auprc_arr[line_idx], 3)}'
                    f', mean={np.mean(auprc_arr[line_idx]):.3f}')

        if curves:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            baseline = list(curves.values())[0]['baseline']
            for tag, c in curves.items():
                axes[0].plot(c['fpr'], c['tpr'],
                             label=f'{tag} - AUROC: {c["auroc"]:.4f}')
                axes[1].plot(c['recall'], c['precision'],
                             label=f'{tag} - AUPRC: {c["auprc"]:.4f}')
            axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8,
                         label='Random - AUROC: 0.5000')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title(f'Pixel-wise ROC - {category} (RDAE variants)')
            axes[0].legend(loc='lower right')
            axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=0.8,
                            label=f'Random - AUPRC: {baseline:.4f}')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title(f'Pixel-wise PRC - {category} (RDAE variants)')
            axes[1].legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(f'temp_var\\roc_prc_{category}_rdae_variants.png', dpi=150)
            plt.show()

    # Combined across all evaluated datasets (original sim only, not contam)
    if eval_contam_levels:
        log_file.close()
        exit()
    log(f'\n--- RDAE Variants {combined_log_label} ---')
    var_sim_curves = {}
    for tag in var_tags:
        if not var_comb_scores[tag]:
            continue
        scores = np.concatenate(var_comb_scores[tag])
        labels = np.concatenate(var_comb_labels[tag])

        fpr, tpr, _          = sklearn.metrics.roc_curve(labels, scores)
        pooled_auroc         = sklearn.metrics.auc(fpr, tpr)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, scores)
        pooled_auprc         = sklearn.metrics.auc(recall, precision)

        var_sim_curves[tag] = {
            'fpr': fpr, 'tpr': tpr, 'auroc': pooled_auroc,
            'recall': recall, 'precision': precision, 'auprc': pooled_auprc,
            'baseline': labels.mean(),
        }
        log(f'{tag:<14}  combined  '
            f'AUROC(pooled): {pooled_auroc:.4f}  AUPRC(pooled): {pooled_auprc:.4f}')

    if var_sim_curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        baseline = list(var_sim_curves.values())[0]['baseline']
        for tag, c in var_sim_curves.items():
            axes[0].plot(c['fpr'], c['tpr'],
                         label=f'{tag} - AUROC: {c["auroc"]:.4f}')
            axes[1].plot(c['recall'], c['precision'],
                         label=f'{tag} - AUPRC: {c["auprc"]:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8,
                     label='Random - AUROC: 0.5000')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(combined_roc_title)
        axes[0].legend(loc='lower right')
        axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=0.8,
                        label=f'Random - AUPRC: {baseline:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(combined_prc_title)
        axes[1].legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(combined_save, dpi=150)
        plt.show()

log_file.close()
