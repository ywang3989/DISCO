from utlis import *
from models import *
# from sklearn.model_selection import train_test_split
# from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
# from torchmetrics.image import StructuralSimilarityIndexMeasure


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
dataset = 'btad_sim'  # 'btad' | 'btad_sim'
if dataset == 'btad':
    data_path = 'BTech_Dataset_Transformed'
    category = 'BTAD3_sys'
    weight_cate = 'BTAD3_sys'
    im_shape = 128
    batch_size = 184
elif dataset == 'btad_sim':
    contamination_level = None
    if contamination_level is None:
        data_path = 'btad_simulation\\02_sim_contam_level_0.02'
        category = '02_sim1'  # '02_sim1' | '02_sim2' | '02_sim3'
    else:
        data_path = 'btad_simulation'
        category = '02_sim_contam_level_' + str(contamination_level)
    weight_cate = '02_sim_1'
    im_shape = 128
    batch_size = 100
root_path = 'data\\' + data_path
os.makedirs('temp_var\\' + category, exist_ok=True)

model_running = 'disco'  # 'disco' | 'disco-wo-p' | 'disco-wo-e' | 'disco-wo-ep' | 'rdae' | 'rpca' | 'memae'

channel_input = 3
kernel_size = 5
lambda1 = 0.05  # Sparsity, lambda
lambda2 = 0.5  # Noise, mu
lambda3 = 3  # Low rank, gamma

lambda4 = 0.0002  # Entropy, only for MemAE
memory_size = 500
threshold = 0.0001 

pretrain_epoches = 100  # 100 for 02_sim, 200 for BTAD3_sys, 500 for MemAE in 02_sim, 2000 for MemAE in BTAD3_sys
epochs_training = int(pretrain_epoches + 100)
loss_print_epoch_step = 10
learning_rate_NN = 0.01
adam_weight_decay = 1e-6
scheduler_factor_NN = 0.5
scheduler_patience_NN = 30

epoches_theta_update = 100
loss_print_epoch_step_theta = 20

epoches_L_update = 200
loss_print_epoch_step_L = 50
learning_rate_L = 5
scheduler_factor_L = 0.1
scheduler_patience_L = 50

rho = 0.8
alpha = 1.05
rho_max = 1e10
epsilon = 5e-3

whether_pretrain = True
whether_continue = False
if whether_pretrain:
    epoch_training_start = pretrain_epoches
else:
    epoch_training_start = 0
if whether_continue:
    epoch_training_start = 104  # resume from the checkpoint saved at epoch 103

## Loading & shuffling data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = MVTEC(root=root_path, train=True, transform=transform, resize=im_shape, category=category)
# train_idx, valid_idx = train_test_split(list(range(len(train_set))), test_size=0.125, shuffle=True)
train_num = len(train_set)
# train_index_shuffled = random.sample(list(range(train_num)), train_num)
# train_sampler = SubsetRandomSampler(train_index_shuffled)
# train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

X = torch.zeros(train_num, 3, im_shape, im_shape).to(device)
L = torch.zeros(train_num, 3, im_shape, im_shape).to(device)
S = torch.zeros(train_num, 3, im_shape, im_shape).to(device)
E = torch.zeros(train_num, 3, im_shape, im_shape).to(device)
Y = torch.zeros(train_num, 3, im_shape, im_shape).to(device)
if im_shape == 256:
    P = torch.zeros(int(np.ceil(train_num/batch_size)), 961, 128, batch_size)  # 31*31
elif im_shape == 128:
    P = torch.zeros(int(np.ceil(train_num/batch_size)), 225, 128, batch_size)  # 15*15
for _, (X_train, _, indices) in enumerate(train_loader):
    X[indices, :, :, :] = X_train.to(device)
X_norm = torch.norm(X).item()
LS = X

## Initialization
if model_running == 'rpca':
    ## Code from https://gist.github.com/jcreinhold/ebf27f997f4c93c2f637c3c900d6388f
    print('==================================================================================================')
    print('Runing Robust PCA')
    for j in range(3):
        x = X[:, j, :, :].reshape(train_num, -1)
        model = RPCA_gpu(D=x, lmbda=lambda1)
        l, s = model.fit()
        l = l.reshape(train_num, im_shape, im_shape)
        s = s.reshape(train_num, im_shape, im_shape)
        L[:, j, :, :] = l
        S[:, j, :, :] = s

    torch.save(L, 'temp_var\\' + category + '\\L_' + category + '_' + model_running + '.pt')
    torch.save(S, 'temp_var\\' + category + '\\S_' + category + '_' + model_running + '.pt')
    print('Completed!')
    print('==================================================================================================')
    quit()
elif model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae']:
    model = LRAE_2d(chnum_in=channel_input, kernel_size=kernel_size).to(device)
elif model_running == 'memae':
    model = MemAE_2d(chnum_in=channel_input, kernel_size=kernel_size, mem_dim=memory_size, 
                     shrink_thres=threshold, low_rank=128, low_rank_processing=False).to(device)

if (not whether_continue) and whether_pretrain:
    pretrained_pth_name = 'temp_var\\pretrained_weight\\weight_' + weight_cate + '_' + str(kernel_size) + '_' + str(epoch_training_start) + '.pth'
    model.load_state_dict(torch.load(pretrained_pth_name))
loss_etrpy = EntropyLossEncap().to(device)
soft_thre = SoftThresholding().to(device)
optimizer_NN = torch.optim.Adam(model.parameters(), lr=learning_rate_NN, weight_decay=adam_weight_decay)
scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, 'min', factor=scheduler_factor_NN, 
                                                          patience=scheduler_patience_NN)
batch_start_L = torch.arange(0, L.shape[0], batch_size)

## Training
print('==================================================================================================')
print(f'Pretraining for {pretrain_epoches:d} epoches. ADMM iteration for ' + model_running)
print('--------------------------------------------------------------------------------------------------')

loss_train_list = []
recon_loss_train_list = []
spars_loss_train_list = []
noise_loss_train_list = []
lrank_loss_train_list = []
etrpy_loss_train_list = []
for epoch_training in range(epoch_training_start, epochs_training):
    recon_loss_train_val = 0
    spars_loss_train_val = 0
    noise_loss_train_val = 0
    lrank_loss_train_val = 0
    etrpy_loss_train_val = 0

    if epoch_training < pretrain_epoches:  # Pretaining
        if model_running == 'memae':
            model.train()
            for i, (X_train, _, _) in enumerate(train_loader):
                X_train = X_train.to(device)
                X_train_recon, att_weight_train = model(X_train)

                recon_loss_train = torch.norm(X_train_recon - X_train)**2 / X_train.shape[0]
                etrpy_loss_train = loss_etrpy(att_weight_train)
                loss_train = recon_loss_train + lambda4 * etrpy_loss_train

                recon_loss_train_val += recon_loss_train.item() * X_train.shape[0]
                etrpy_loss_train_val += etrpy_loss_train.item() * X_train.shape[0]

                optimizer_NN.zero_grad()
                loss_train.backward()
                optimizer_NN.step()

            recon_loss_train_val /= train_num
            lrank_loss_train_val /= train_num
            loss_train_val = recon_loss_train_val + lambda4 * etrpy_loss_train_val

            scheduler_NN.step(loss_train_val)
        
        elif model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae']:
            # model.train()
            for i, (X_train, _, _) in enumerate(train_loader):
                X_train = X_train.to(device)
                X_train_recon, position_matrix = model(X_train)
                P[i, :, :, :] = position_matrix
                
                # mse_recon_loss_train = loss_recon_mse(X_train_recon, X_train)
                recon_loss_train = torch.norm(X_train_recon - X_train)**2 / X_train.shape[0]
                lrank_loss_train = torch.mean(torch.norm(position_matrix, 'nuc', dim=(1, 2)))
                loss_train = recon_loss_train + lambda3 * lrank_loss_train
                
                recon_loss_train_val += recon_loss_train.item() * X_train.shape[0]
                lrank_loss_train_val += lrank_loss_train.item() * X_train.shape[0]

                optimizer_NN.zero_grad()
                loss_train.backward()
                optimizer_NN.step()

            recon_loss_train_val /= train_num
            lrank_loss_train_val /= train_num
            loss_train_val = recon_loss_train_val + lambda3 * lrank_loss_train_val

            scheduler_NN.step(loss_train_val)
        
    else:  # ADMM
        loss_print_epoch_step = 1
    
        if whether_continue and (epoch_training == epoch_training_start):
            model.load_state_dict(torch.load('temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training) + '.pth'))
            rho = 0.8 * 1.05**(epoch_training_start - pretrain_epoches)
            L = torch.load('temp_var\\' + category + '\\L_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training) + '.pt')
            S = torch.load('temp_var\\' + category + '\\S_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training) + '.pt')
            E = torch.load('temp_var\\' + category + '\\E_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training) + '.pt')
            Y = torch.load('temp_var\\' + category + '\\Y_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training) + '.pt')
            LS = L + S + E
            print(f'Previous ({epoch_training}) Epoch Constraint Error (e1): {(torch.norm(X-L-S-E).item() / X_norm):.8f}')

        print(f'ADMM - Epoch: {epoch_training+1} & rho: {rho}')

        if model_running == 'memae':
            print('No ADMM for MemAE')
            quit()

        elif model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep']:
            # Updating L
            print('--------------------------------------------------------------------------------------------------')
            print('    > Updating L')
            model.eval()        
            L.requires_grad = True
            optimizer_L = torch.optim.SGD([L], lr=learning_rate_L)
            scheduler_L = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_L, 'min', factor=scheduler_factor_L, 
                                                                    patience=scheduler_patience_L)
            for epoch_L_update in range(epoches_L_update):
                f, p = model(L)
                recon_loss_L = torch.norm(f-L)**2 / train_num
                lrank_loss_L = torch.mean(torch.norm(p, 'nuc', dim=(1, 2)))
                costr_loss_L = rho/2 * torch.norm(X - L - S - E + Y/rho)**2 / train_num
                loss_L = recon_loss_L + lambda3 * lrank_loss_L + costr_loss_L

                optimizer_L.zero_grad()
                loss_L.backward()
                optimizer_L.step()

                scheduler_L.step(loss_L.item())
                if epoch_L_update % loss_print_epoch_step_L == 0:
                    print('        ----------------------------------------------------')
                    print(f'        L Updating Iteration: {epoch_L_update:d}, LR: {optimizer_L.param_groups[0]["lr"]:.2e}')
                    print(f'        Recon: {recon_loss_L.item():.4f}, Constraint: {costr_loss_L.item():.4f}, Lrank: {lrank_loss_L.item():.4f}')

            # Updating S, E, Y, rho
            print('--------------------------------------------------------------------------------------------------')
            print('    > Updating S, E, Y, rho')
            with torch.no_grad():
                S = soft_thre(X - L - E + Y/rho, lambda1/rho)
                if model_running in ['disco', 'disco-wo-p']:
                    E = rho / (2*lambda2 + rho) * (X - L - S + Y/rho)
                Y = Y + rho*(X - L - S - E)
                rho = np.min([alpha*rho, rho_max])
            spars_loss_train_val = torch.norm(S, p=1).item() / train_num
            noise_loss_train_val = torch.norm(E).item() ** 2 / train_num
            print(f'        Spars: {spars_loss_train_val:.4f}, Noise: {noise_loss_train_val:.4f}')

        elif model_running == 'rdae':
            # Updating L & S
            print('--------------------------------------------------------------------------------------------------')
            print('    > Updating L & S')
            model.eval()
            if epoch_training == pretrain_epoches:
                L, _ = model(X)
            else:
                L, _ = model(L)

            with torch.no_grad():
                S = soft_thre(X - L, lambda1)
            spars_loss_train_val = torch.norm(S, p=1).item() / train_num
            print(f'        Spars: {spars_loss_train_val:.4f}')

        # Updating theta
        print('--------------------------------------------------------------------------------------------------')
        print('    > Updating theta')
        model.train()
        learning_rate_NN = 0.001
        optimizer_NN = torch.optim.Adam(model.parameters(), lr=learning_rate_NN, weight_decay=adam_weight_decay)
        scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, 'min', factor=scheduler_factor_NN, 
                                                                  patience=scheduler_patience_NN)
        # L_label = L.detach().to(device)
        L_perm = L[torch.randperm(L.shape[0])].detach()
        for epoch_theta_update in range(epoches_theta_update):
            # for _, (X_train, _, indices) in enumerate(train_loader):
            for start in batch_start_L:
                # X_train = X_train.to(device)
                L_train = L_perm[start:start+batch_size, :].to(device)
                L_train_recon, position_matrix = model(L_train)
                P[int(start/batch_size), :, :, :] = position_matrix

                # recon_loss_train = loss_recon_mse(L_train_recon, L_label[indices])
                recon_loss_train = torch.norm(L_train_recon - L_train)**2 / L_train.shape[0]
                lrank_loss_train = torch.mean(torch.norm(position_matrix, 'nuc', dim=(1, 2)))
                if model_running == 'rdae':
                    lrank_loss_train = torch.tensor(0)
                loss_train = recon_loss_train + lambda3 * lrank_loss_train

                recon_loss_train_val += recon_loss_train.item() * L_train.shape[0]
                lrank_loss_train_val += lrank_loss_train.item() * L_train.shape[0]

                optimizer_NN.zero_grad()
                loss_train.backward()
                optimizer_NN.step()

            scheduler_NN.step(loss_train.item())

            recon_loss_train_val /= train_num
            lrank_loss_train_val /= train_num

            if (epoch_theta_update + 1) % loss_print_epoch_step_theta == 0:
                print('        ----------------------------------------------------')
                print(f'        Theta Updating Iteration: {(epoch_theta_update+1):d}, LR: {optimizer_NN.param_groups[0]["lr"]:.2e}')
                print(f'        Recon: {recon_loss_train_val:.4f}, Lrank: {lrank_loss_train_val:.4f}')

        torch.save(model.state_dict(), 'temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pth')
        torch.save(L, 'temp_var\\' + category + '\\L_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pt')
        torch.save(S, 'temp_var\\' + category + '\\S_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pt')
        torch.save(E, 'temp_var\\' + category + '\\E_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pt')
        torch.save(Y, 'temp_var\\' + category + '\\Y_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pt')
    
    ## Calculating errors
    e1 = torch.norm(X - L - S - E).item() / X_norm
    e2 = torch.norm(LS - L - S - E).item() / X_norm

    ## Recoding loss
    recon_loss_train_list.append(recon_loss_train_val)
    spars_loss_train_list.append(spars_loss_train_val)
    noise_loss_train_list.append(noise_loss_train_val)
    lrank_loss_train_list.append(lrank_loss_train_val)
    etrpy_loss_train_list.append(etrpy_loss_train_val)
    loss_train_val = recon_loss_train_val + lambda1 * spars_loss_train_val + lambda2 * noise_loss_train_val + lambda3 * lrank_loss_train_val + lambda4 * etrpy_loss_train_val
    loss_train_list.append(loss_train_val)
    
    ## Printing training info
    if (epoch_training + 1) % loss_print_epoch_step == 0:
        print('--------------------------------------------------------------------------------------------------')
        print(f'Epoch: {(epoch_training+1):d}, LR: {optimizer_NN.param_groups[0]["lr"]:.2e}')
        print(f'    Training Loss: {loss_train_val:.4f}')
        print(f'        Recon: {recon_loss_train_val:.4f}, Spars: {spars_loss_train_val:.4f}')
        print(f'        Noise: {noise_loss_train_val:.4f}, Lrank: {lrank_loss_train_val:.4f}')
        print(f'            For MemAE: Etrpy {etrpy_loss_train_val:.4f}')
        print(f'    Constraint Error (e1): {e1:.8f}, Fixed-Point Error (e2): {e2:.8f}')
        if model_running in ['aslr', 'mosrdae', 'mordae', 'rdae']:
            positional_rank = torch.zeros(P.shape[0], P.shape[1])
            for i in range(P.shape[0]):
                positional_rank[i, :] = torch.linalg.matrix_rank(P[i, :, :, :]).float()
            positional_rank = torch.mean(positional_rank, dim=0)
            print(f'    Positional Rank: avg {torch.mean(positional_rank).item():.4f}, std {torch.std(positional_rank).item():.4f}')
            print(f'                     min {torch.min(positional_rank).item():.4f}, max {torch.max(positional_rank).item():.4f}')
            print('==================================================================================================')

    ## Convergence checking
    if (e1 < epsilon or e2 < epsilon) and (epoch_training >= pretrain_epoches):
        print('Converged!')
        break

    ## Updating LS
    LS = L + S + E

    ## Saving temp weight
    if (epoch_training + 1) % 100 == 0:
        temp_pth_name = 'temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pth'
        torch.save(model.state_dict(), temp_pth_name)

# torch.save(model.state_dict(), 'model_para\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(epoch_training+1) + '.pth')

## Ploting loss
plt.plot(loss_train_list, label='Total Training Loss')
plt.plot(recon_loss_train_list, label='Reconstruction Training Loss')
plt.plot(spars_loss_train_list, label='Sparsity Training Loss')
plt.plot(noise_loss_train_list, label='Noise Training Loss')
plt.plot(lrank_loss_train_list, label='Low-rank Training Loss')
if model_running == 'memae':
    plt.plot(etrpy_loss_train_list, label='Entropy Training Loss (MemAE)')
plt.yscale('log')
plt.legend()
plt.show()
