from utlis import *
from models import *
from torch.utils.data import DataLoader
from matplotlib import colors
import torchvision
import sklearn


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

## Hyperparameters
dataset = 'btad_sim'
if dataset == 'btad':
    data_path = 'BTech_Dataset_Transformed'
    category = 'BTAD3_sys'
    im_shape = 128
elif dataset == 'btad_sim':
    contamination_level = None
    if contamination_level is None:
        data_path = 'btad_simulation\\02_sim_contam_level_0.02'
        category = '02_sim3'  # '02_sim1' | '02_sim2' | '02_sim3'
    else:
        data_path = 'btad_simulation'
        category = '02_sim_contam_level_' + str(contamination_level)
    im_shape = 128
root_path = 'data\\' + data_path
S_gtruth_path = root_path + '\\' + category + '\\train\\defect_ground_truth\\'
L_gtruth_path = root_path + '\\' + category + '\\train\\defect_background\\'

batch_size = 64
channel_input = 3
kernel_size = 5
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model_running = 'disco'  # 'disco' | 'disco-wo-p' | 'disco-wo-e' | 'disco-wo-ep' | 'rdae' | 'rpca' | 'memae'
model_para = {
    '02_sim1': {
        'disco': [106, 0.14],  # [104, 0.15]
        'disco-wo-p': [105, 0.06],
        'disco-wo-e': [105, 0.23],
        'disco-wo-ep': [108, 0.07],
        'rdae': [103, 0.08],
        'rpca': [0, 0.01],
        'memae': [500, 0.14],
        },
    '02_sim2': {
        'disco': [104, 0.17], 
        'disco-wo-p': [105, 0.07], 
        'disco-wo-e': [105, 0.25], 
        'disco-wo-ep': [108, 0.09], 
        'rdae': [103, 0.11], 
        'rpca': [0, 0.01], 
        'memae': [500, 0.13]
        },
    '02_sim3': {
        'disco': [104, 0.16], 
        'disco-wo-p': [105, 0.04], 
        'disco-wo-e': [104, 0.22], 
        'disco-wo-ep': [107, 0.08], 
        'rdae': [103, 0.17], 
        'rpca': [0, 0.01], 
        'memae': [500, 0.13]
        },
    'BTAD3_sys': {
        'disco': [213, 0.21], 
        'disco-wo-p': [209, 0.15], 
        'disco-wo-e': [214, 0.43], 
        'disco-wo-ep': [212, 0.47], 
        'rdae': [203, 0.40], 
        'rpca': [0, 0.01], 
        'memae': [2000, 0.32]
        },
    '02_sim_contam_level_0.05': {
        'disco': [105, 0.16],
        'rdae': [105, 0.26], 
        'rpca': [0, 0.01], 
        'memae': [500, 0.13]
        },
    '02_sim_contam_level_0.08': {
        'disco': [105, 0.16], 
        'rdae': [106, 0.24], 
        'rpca': [0, 0.01], 
        'memae': [500, 0.12]
        },
    '02_sim_contam_level_0.1': {
        'disco': [104, 0.16], 
        'rdae': [105, 0.25], 
        'rpca': [0, 0.01], 
        'memae': [500, 0.12]
        }
    }
training_epoch = model_para[category][model_running][0]
thre_best = model_para[category][model_running][1]
thre_upper = 2

whether_plot = True
whether_just_result = True

memory_size = 500
threshold = 0.0001

to_see_pretrain = False
pretraining_epoch = 200

to_see_train = True

## Loading data
train_set_defect = MVTEC(root=root_path, train=True, transform=transform, resize=im_shape, category=category, train_defect=True)
train_loader_defect = DataLoader(train_set_defect, batch_size=batch_size, shuffle=False)
# test_set = MVTEC(root=root_path, train=False, transform=transform, resize=im_shape, category=category)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

## Loading model & result
if model_running == 'rpca':
    L_train = torch.load('temp_var\\' + category + '\\L_' + category + '_' + model_running + '.pt')
    S_train = torch.load('temp_var\\' + category + '\\S_' + category + '_' + model_running + '.pt')
elif model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae', 'memae']:
    if model_running == 'memae':
        model = MemAE_2d(chnum_in=channel_input, kernel_size=kernel_size, mem_dim=memory_size, 
                     shrink_thres=threshold, low_rank=128, low_rank_processing=False).to(device)
    else:
        model = LRAE_2d(chnum_in=channel_input, kernel_size=kernel_size).to(device)
        L_train = torch.load('temp_var\\' + category + '\\L_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(training_epoch) + '.pt')
        S_train = torch.load('temp_var\\' + category + '\\S_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(training_epoch) + '.pt')
        E_train = torch.load('temp_var\\' + category + '\\E_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(training_epoch) + '.pt')
    pth_name = 'temp_var\\' + category + '\\weight_' + category + '_' + model_running + '_' + str(kernel_size) + '_' + str(training_epoch) + '.pth'
    model.eval()
    model.load_state_dict(torch.load(pth_name))

if to_see_pretrain:
    pth_name = 'temp_var\\pretrained_weight\\weight_' + category + '_' + str(kernel_size) + '_' + str(pretraining_epoch) + '.pth'
    model = LRAE_2d(chnum_in=channel_input, kernel_size=kernel_size).to(device)
    model.eval()
    model.load_state_dict(torch.load(pth_name))

## Visualization
threholds = np.linspace(0.01, thre_upper, int(100*thre_upper))
metrics_thre = np.zeros((int(100*thre_upper), 8))
resizeTransf = transforms.Resize(im_shape, 2, antialias=True)

if to_see_train:  # For defective training samples
    for idx, thre in enumerate(threholds):
        # Ploting
        dice_coefs = []
        auroc_vals = []
        bkgd_psnrs = []
        bkgd_ssims = []
        for batch_idx, (X_defect, _, indices) in enumerate(train_loader_defect):
            for i in range(X_defect.shape[0]):
                X_image_show = X_defect[i].permute(1, 2, 0).cpu().detach().numpy()
                if model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae', 'rpca']:
                    L_backgd = L_train[indices[i]].view(1, channel_input, im_shape, im_shape).cpu().detach()
                    L_backgd_show = np.clip(L_backgd.squeeze().permute(1, 2, 0).cpu().detach().numpy(), -1., 1.)
                    S_defect_show = np.sum(np.abs(S_train[indices[i]].permute(1, 2, 0).cpu().detach().numpy()), axis=2)
                    if model_running == 'rpca':
                        E_noise_show = np.zeros((im_shape, im_shape, 1))
                    elif model_running in ['disco', 'disco-wo-p', 'disco-wo-e', 'disco-wo-ep', 'rdae']: 
                        E_noise_show = np.sum(np.abs(E_train[indices[i]].permute(1, 2, 0).cpu().detach().numpy()), axis=2)
                elif model_running == 'memae':
                    X_input = X_defect[i].view(1, channel_input, im_shape, im_shape).to(device)
                    L_backgd, _ = model(X_input)
                    L_backgd_show = np.clip(L_backgd.squeeze().permute(1, 2, 0).cpu().detach().numpy(), -1., 1.)
                    S_defect = X_input - L_backgd
                    S_defect_show = np.sum(np.abs(S_defect.squeeze().permute(1, 2, 0).cpu().detach().numpy()), axis=2)
                    E_noise_show = np.zeros((im_shape, im_shape, 1))

                if to_see_pretrain:
                    L_recon, _ = model(X_defect[i].view(1, channel_input, im_shape, im_shape).to(device))
                    L_backgd_show = np.clip(L_recon.squeeze().permute(1, 2, 0).cpu().detach().numpy(), -1., 1.)
                
                # Anomaly detection
                S_ground_truth = torchvision.io.read_image(S_gtruth_path + str(indices[i].item()) + '.png')
                S_ground_truth = resizeTransf(S_ground_truth).squeeze().detach().numpy()
                S_ground_truth[S_ground_truth != 0] = 1
                if S_ground_truth.shape[0] == 4:
                    S_ground_truth = S_ground_truth[0, :, :]
                if whether_plot:
                    S_mask_pred = (S_defect_show > thre_best).astype(int)
                else:
                    S_mask_pred = (S_defect_show > thre).astype(int)
                dice_coef = dice_coefficient(S_ground_truth, S_mask_pred)
                auroc_val = sklearn.metrics.roc_auc_score(S_ground_truth.flatten(), S_mask_pred.flatten())
                dice_coefs.append(dice_coef)
                auroc_vals.append(auroc_val)

                # Background restoration
                if dataset == 'btad_sim':
                    L_ground_truth_show = mpimg.imread(L_gtruth_path + str(indices[i].item()) + '.png')
                elif dataset == 'btad':
                    L_ground_truth_show = np.zeros((im_shape, im_shape, 3))
                if L_ground_truth_show.shape[2] == 4:
                    L_ground_truth_show = L_ground_truth_show[:, :, 0:3]
                L_ground_truth = Image.fromarray((L_ground_truth_show * 255).astype(np.uint8))
                L_ground_truth = transform(resizeTransf(L_ground_truth))

                K = 20
                interpol_pts = np.linspace(0, 1, num=K+1)
                L_ground_truth_ = (L_ground_truth * 127.5 + 127.5).squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.int16)
                L_backgd_ = (L_backgd * 127.5 + 127.5).squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.int16)
                if dataset == 'btad_sim':
                    psnrs, dms_psnr = DMS_f(S_ground_truth, interpol_pts, peak_signal_to_noise_ratio, L_ground_truth_, L_backgd_)
                    ssims, dms_ssim = DMS_f(S_ground_truth, interpol_pts, structural_similarity, L_ground_truth_, L_backgd_)
                elif dataset == 'btad':
                    dms_psnr, dms_ssim = 0., 0.
                bkgd_psnrs.append(dms_psnr)
                bkgd_ssims.append(dms_ssim)

                if whether_plot and not whether_just_result:
                    # print('---------------------')
                    # print('DMS_' + str(K) + '-PSNR: ', psnrs)
                    # print('DMS_' + str(K) + '-SSIM: ', ssims)
                    # np.savetxt('dms' + str(K) + '_' + model_running + '_02_sim2_' + str(i+1) + '.csv', np.concatenate((np.expand_dims(psnrs, 0), np.expand_dims(ssims, 0)), 0), delimiter=',')
                    if dataset == 'btad_sim':
                        plot_dms_f(S_ground_truth, interpol_pts, X_image_show, L_backgd_show, L_ground_truth_show, 
                                [psnrs, dms_psnr, ssims, dms_ssim])
                    plot_result(X_image_show, L_backgd_show, L_ground_truth_show, S_defect_show, E_noise_show, 
                                S_mask_pred, S_ground_truth, [dice_coef, auroc_val, dms_psnr, dms_ssim])

            if whether_plot and whether_just_result:
                print(model_running + ' on ' + category)
                print(f'    Dice: {np.round(dice_coefs, 3)} | mean: {np.round(np.mean(dice_coefs), 3)}, std: {np.round(np.std(dice_coefs, ddof=1), 3)}')
                if dataset == 'btad_sim':
                    print(f'   AUROC: {np.round(auroc_vals, 3)} | mean: {np.round(np.mean(auroc_vals), 3)}, std: {np.round(np.std(auroc_vals, ddof=1), 3)}')
                    print(f'    PSNR: {np.round(bkgd_psnrs, 3)} | mean: {np.round(np.mean(bkgd_psnrs), 3)}, std: {np.round(np.std(bkgd_psnrs, ddof=1), 3)}')
                    print(f'    SSIM: {np.round(bkgd_ssims, 3)} | mean: {np.round(np.mean(bkgd_ssims), 3)}, std: {np.round(np.std(bkgd_ssims, ddof=1), 3)}')
                quit()

        if whether_plot and dataset == 'btad_sim':
            if category == '02_sim1':
                blob_index = [0, 2, 4, 6]
                line_index = [1, 3, 5, 7]
            elif category == '02_sim2':
                blob_index = [0, 2, 3, 4]
                line_index = [1, 5, 6, 7]
            elif category == '02_sim3':
                blob_index = [1, 3, 6, 7]
                line_index = [0, 2, 4, 5]
            print(f'Group Dice Avg: {np.round(np.mean(np.array(dice_coefs)[blob_index]), 3), np.round(np.mean(np.array(dice_coefs)[line_index]), 3)}')
            print(f'           Std: {np.round(np.std(np.array(dice_coefs)[blob_index], ddof=1), 3), np.round(np.std(np.array(dice_coefs)[line_index], ddof=1), 3)}')
        
        if whether_plot and dataset == 'btad':
            dice_coefs_group = np.reshape(np.array(dice_coefs), (5, 4))
            dice_coef_wo_spot_avg, dice_coef_wo_spot_std = np.round(np.mean(np.array(dice_coefs[:15])), 3), np.round(np.std(np.array(dice_coefs[:15]), ddof=1), 3)
            auroc_vals_group = np.reshape(np.array(auroc_vals), (5, 4))
            auroc_val_wo_spot_avg, auroc_val_wo_spot_std = np.round(np.mean(np.array(auroc_vals[:15])), 3), np.round(np.std(np.array(auroc_vals[:15]), ddof=1), 3)
            print(f'Group Dice Avg: {np.round(np.mean(dice_coefs_group, axis=1), 3)}')
            print(f'           Std: {np.round(np.std(dice_coefs_group, axis=1, ddof=1), 3)}')
            print(f'Group AUROC Avg: {np.round(np.mean(auroc_vals_group, axis=1), 3)}')
            print(f'            Std: {np.round(np.std(auroc_vals_group, axis=1, ddof=1), 3)}')
            quit()

        metrics_thre[idx, 0] = np.mean(dice_coefs)
        metrics_thre[idx, 1] = np.std(dice_coefs, ddof=1)
        metrics_thre[idx, 2] = np.mean(auroc_vals)
        metrics_thre[idx, 3] = np.std(auroc_vals, ddof=1)
        metrics_thre[idx, 4] = np.mean(bkgd_psnrs)
        metrics_thre[idx, 5] = np.std(bkgd_psnrs, ddof=1)
        metrics_thre[idx, 6] = np.mean(bkgd_ssims)
        metrics_thre[idx, 7] = np.std(bkgd_ssims, ddof=1)

    max_idx = np.argmax(metrics_thre[:, 0])
    plt.plot(threholds, metrics_thre, label=['Dice Avg: ' + str(np.round(metrics_thre[max_idx, 0], 3)), 
                                             'Dice Std: ' + str(np.round(metrics_thre[max_idx, 1], 3)), 
                                             'AUROC Avg: ' + str(np.round(metrics_thre[max_idx, 2], 3)), 
                                             'AUROC Std: ' + str(np.round(metrics_thre[max_idx, 3], 3)), 
                                             r'$\mathrm{DMS}_{5}$-PSNR Avg: ' + str(np.round(metrics_thre[max_idx, 4], 3)),
                                             r'$\mathrm{DMS}_{5}$-PSNR Std: ' + str(np.round(metrics_thre[max_idx, 5], 3)),
                                             r'$\mathrm{DMS}_{5}$-SSIM Avg: ' + str(np.round(metrics_thre[max_idx, 6], 3)),
                                             r'$\mathrm{DMS}_{5}$-SSIM Std: ' + str(np.round(metrics_thre[max_idx, 7], 3))])
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 0], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 1], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 2], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 3], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 4], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 5], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 6], marker='x', color='cyan')
    plt.plot(threholds[max_idx], metrics_thre[max_idx, 7], marker='x', color='cyan')
    plt.text(threholds[max_idx] + 0.02, metrics_thre[max_idx, 0], 'Thre: ' + str(np.round(threholds[max_idx], 3)))
    plt.xlabel("Threshold")
    plt.title('#' + str(training_epoch) + ' of ' + model_running + ' on ' + category)
    plt.yscale('log')
    plt.legend()
    plt.show()
else:  # For testing samples
    for batch_idx, (X_test, _, _) in enumerate(train_loader_defect):
        X_test = X_test.to(device)
        L_test_recon, _ = model(X_test)
        S_test = torch.abs(torch.sub(X_test, L_test_recon))

        images_test_show = X_test.permute(0, 2, 3, 1).cpu().detach().numpy()
        images_test_recon_show = L_test_recon.permute(0, 2, 3, 1).cpu().detach().numpy()
        anomaly_score_map_show = S_test.permute(0, 2, 3, 1).cpu().detach().numpy()
        for i in range(X_test.shape[0]):
            plt.style.use('classic')
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow((images_test_show[i, :, :, :] * 127.5 + 127.5).astype(np.uint8))
            axs[0].axis('off')
            axs[0].set_title('X')
            axs[1].imshow((images_test_recon_show[i, :, :, :] * 127.5 + 127.5).astype(np.uint8))
            axs[1].axis('off')
            axs[1].set_title('L_recon')
            axs[2].imshow(np.max(anomaly_score_map_show[i, :, :, :], axis=2))
            axs[2].axis('off')
            axs[2].set_title('max|X-L_recon|')
            norm = colors.Normalize(0, np.max(np.max(anomaly_score_map_show[i, :, :, :], axis=2)))
            fig.colorbar(cm.ScalarMappable(norm=norm), ax=axs, shrink=0.5)
            plt.show()

