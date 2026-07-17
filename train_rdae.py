import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utlis import MVTEC, SoftThresholding

## ── Config ────────────────────────────────────────────────────────────────────
category = '02_sim_contam_level_0.1'

# (s1, s2, s3) per encoder layer
# (1, 2, 2) -> 30 x 30
# (1, 2, 3) -> 20 x 20
# (2, 2, 2) -> 15 x 15
# (2, 3, 4) ->  5 x  5
# (3, 3, 4) ->  3 x  3
strides = (2, 3, 4)


## ── Hyperparameters ───────────────────────────────────────────────────────────
kernel_size           = 5
channel_input         = 3
lambda1               = 0.05
im_shape              = 128
admm_epochs           = 100
lr_pretrain           = 0.01
lr_admm               = 0.001
adam_weight_decay     = 1e-6
scheduler_factor      = 0.5
scheduler_patience    = 30
epochs_theta_update   = 100
loss_print_step       = 10
loss_print_step_theta = 20
epsilon               = 5e-3

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
pretrain_epoch_map = {
    '02_sim1':                  100,
    '02_sim2':                  100,
    '02_sim3':                  100,
    'BTAD3_sys':                200,
    '02_sim_contam_level_0.05': 100,
    '02_sim_contam_level_0.08': 100,
    '02_sim_contam_level_0.1':  100,
}

batch_size   = batch_size_map[category]
root_path    = 'data\\' + data_path_map[category]
stride_tag   = ''.join(map(str, strides))
model_tag    = f'rdae_s{stride_tag}'
pretrain_epochs = pretrain_epoch_map[category]
total_epochs = pretrain_epochs + admm_epochs

out_dir = f'temp_var\\rdae_bottleneck_variants\\{category}'
os.makedirs(out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}  |  category: {category}  |  model: {model_tag}')

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

        # Compute intermediate spatial sizes via dummy forward
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
        print(f'  Spatial: {im_shape} → {H1} → {H2} → {H3}  |  decoder output_padding: ({op1}, {op2}, {op3})')

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


## ── Data loading ──────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set    = MVTEC(root=root_path, train=True, transform=transform,
                     resize=im_shape, category=category)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_num    = len(train_set)
print(f'Training set size: {train_num}  |  batch size: {batch_size}')

## ── Pre-fill full X tensor (needed for ADMM) ──────────────────────────────────
X = torch.zeros(train_num, channel_input, im_shape, im_shape).to(device)
L = torch.zeros(train_num, channel_input, im_shape, im_shape).to(device)
S = torch.zeros(train_num, channel_input, im_shape, im_shape).to(device)
for _, (X_batch, _, indices) in enumerate(train_loader):
    X[indices] = X_batch.to(device)
X_norm = torch.norm(X).item()
LS     = X.clone()

## ── Model + optimizer ─────────────────────────────────────────────────────────
model = LRAE_2d_custom(chnum_in=channel_input, kernel_size=kernel_size,
                       strides=strides, im_shape=im_shape).to(device)
soft_thre    = SoftThresholding().to(device)
optimizer_NN = torch.optim.Adam(model.parameters(), lr=lr_pretrain,
                                weight_decay=adam_weight_decay)
scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_NN, 'min', factor=scheduler_factor, patience=scheduler_patience)
batch_start_L = torch.arange(0, train_num, batch_size)

# Determine spatial size of position matrix for rank tracking
with torch.no_grad():
    _, p_probe = model(X[:1])
spatial_size = p_probe.shape[0]
P = torch.zeros(len(batch_start_L), spatial_size, 128, batch_size)

## ── Training loop ─────────────────────────────────────────────────────────────
print('=' * 90)
print(f'Pretraining {pretrain_epochs} epochs  →  ADMM {admm_epochs} epochs  |  {model_tag}')
print('=' * 90)

loss_list = []
for epoch in range(total_epochs):

    recon_loss_val = 0.0
    spars_loss_val = 0.0

    ## ── Pretraining ───────────────────────────────────────────────────────────
    if epoch < pretrain_epochs:
        model.train()
        for i, (X_batch, _, _) in enumerate(train_loader):
            X_batch    = X_batch.to(device)
            recon, _   = model(X_batch)
            recon_loss = torch.norm(recon - X_batch) ** 2 / X_batch.shape[0]
            recon_loss_val += recon_loss.item() * X_batch.shape[0]

            optimizer_NN.zero_grad()
            recon_loss.backward()
            optimizer_NN.step()

        recon_loss_val /= train_num
        scheduler_NN.step(recon_loss_val)

    ## ── ADMM ──────────────────────────────────────────────────────────────────
    else:
        # ── L & S update ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            if epoch == pretrain_epochs:
                L, _ = model(X)
            else:
                L, _ = model(L)
            S = soft_thre(X - L, lambda1)
        spars_loss_val = torch.norm(S, p=1).item() / train_num

        # ── Theta update (train on L, reconstruction loss only) ───────────────
        model.train()
        optimizer_NN = torch.optim.Adam(model.parameters(), lr=lr_admm,
                                        weight_decay=adam_weight_decay)
        scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_NN, 'min', factor=scheduler_factor, patience=scheduler_patience)
        L_perm = L[torch.randperm(L.shape[0])].detach()

        for theta_ep in range(epochs_theta_update):
            for bi, start in enumerate(batch_start_L):
                L_batch      = L_perm[start:start + batch_size].to(device)
                recon, pos_mat = model(L_batch)
                if bi < P.shape[0]:
                    P[bi, :, :, :P.shape[3]] = pos_mat.detach().cpu()
                recon_loss   = torch.norm(recon - L_batch) ** 2 / L_batch.shape[0]
                loss         = recon_loss  # lrank_loss = 0 for rdae

                recon_loss_val += recon_loss.item() * L_batch.shape[0]

                optimizer_NN.zero_grad()
                loss.backward()
                optimizer_NN.step()

            scheduler_NN.step(recon_loss.item())
            recon_loss_val /= train_num

            if (theta_ep + 1) % loss_print_step_theta == 0:
                print(f'    Theta iter {theta_ep+1:3d}  |  LR: {optimizer_NN.param_groups[0]["lr"]:.2e}'
                      f'  |  Recon: {recon_loss_val:.4f}')

        # Save ADMM checkpoint
        pth = f'{out_dir}\\weight_{category}_{model_tag}_{kernel_size}_{epoch+1}.pth'
        torch.save(model.state_dict(), pth)
        torch.save(L, f'{out_dir}\\L_{category}_{model_tag}_{kernel_size}_{epoch+1}.pt')
        torch.save(S, f'{out_dir}\\S_{category}_{model_tag}_{kernel_size}_{epoch+1}.pt')

    ## ── Logging ───────────────────────────────────────────────────────────────
    e1 = torch.norm(X - L - S).item() / X_norm
    e2 = torch.norm(LS - L - S).item() / X_norm
    loss_val = recon_loss_val + lambda1 * spars_loss_val
    loss_list.append(loss_val)
    LS = L + S

    if (epoch < pretrain_epochs and (epoch + 1) % loss_print_step == 0) or epoch >= pretrain_epochs:
        print('-' * 90)
        print(f'Epoch {epoch+1:3d}  |  LR: {optimizer_NN.param_groups[0]["lr"]:.2e}')
        print(f'  Recon: {recon_loss_val:.4f}  |  Spars: {spars_loss_val:.4f}')
        print(f'  Constraint (e1): {e1:.6f}  |  Fixed-point (e2): {e2:.6f}')
        if epoch >= pretrain_epochs:
            positional_rank = torch.zeros(P.shape[0], P.shape[1])
            for i in range(P.shape[0]):
                positional_rank[i] = torch.linalg.matrix_rank(P[i]).float()
            pr = positional_rank.mean(dim=0)
            print(f'  Positional rank: avg {pr.mean():.2f}, std {pr.std():.2f}, '
                  f'min {pr.min():.2f}, max {pr.max():.2f}')

    ## ── Convergence check ─────────────────────────────────────────────────────
    if (e1 < epsilon or e2 < epsilon) and epoch >= pretrain_epochs:
        print('Converged!')
        break

    ## ── Periodic checkpoint (every 100 epochs, pretraining only) ─────────────
    if (epoch + 1) % 100 == 0 and epoch < pretrain_epochs:
        torch.save(model.state_dict(),
                   f'{out_dir}\\weight_{category}_{model_tag}_{kernel_size}_{epoch+1}.pth')

## ── Final save ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(),
           f'{out_dir}\\weight_{category}_{model_tag}_{kernel_size}_{epoch+1}.pth')
print(f'\nTraining complete. Final epoch: {epoch+1}')

## ── Loss plot ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss — {model_tag} — {category}')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{out_dir}\\loss_{category}_{model_tag}.png', dpi=150)
plt.show()
