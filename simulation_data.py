import os
from utlis import *
from models import *
from torch.utils.data import DataLoader
import random


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_path = 'BTech_Dataset_Transformed'
root_path = 'data\\' + data_path
category = '02'

contamination_level = 0.05

im_shape = 128
channel_input = 3
kernel_size = 5
batch_size = 400
low_rank = 20
anomaly_number = int(batch_size * contamination_level)
color_offset = -30
show_pretrain = False
show_low_rank = False
process_save_low_rank = False
process_save_anomaly = True

if category == 'carpet':
    sim_data = category
    pth_path = 'simulation\\' + sim_data + '\\weight\\weight_' + category + '_aslr_' + str(kernel_size) + '_'
elif category == '02':
    sim_data = 'btad2'
    pth_path = 'simulation\\' + sim_data + '\\weight\\weight_' + category + '_0_' + str(kernel_size) + '_'

train_set_good = MVTEC(root=root_path, train=True, transform=transform, resize=im_shape, category=category)
train_loader = DataLoader(train_set_good, batch_size=batch_size, shuffle=False)
test_set = MVTEC(root=root_path, train=False, transform=transform, resize=im_shape, category=category)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
train_size = train_set_good.train_data.shape[0]

model = LRAE_2d(chnum_in=channel_input, kernel_size=kernel_size).to(device)
decoder = model.decoder
model.eval()

if show_pretrain:
    images_recon_show = np.zeros((15, im_shape, im_shape, channel_input))
    rmse = np.zeros(15)
    for _, (X, _, _) in enumerate(train_loader):
        image_input_show = X[0].permute(1, 2, 0).cpu().detach().numpy()
        for i in range(15):
            pth_name = pth_path + str(int((i+1)*100)) + '.pth'
            model.load_state_dict(torch.load(pth_name))

            image_input = X[0].view(1, channel_input, im_shape, im_shape).to(device)
            image_recon, _ = model(image_input)

            images_recon_show[i] = np.clip(torch.squeeze(image_recon).permute(1, 2, 0).cpu().detach().numpy(), -1., 1.)
            rmse[i] = np.sqrt(nn.functional.mse_loss(image_input, image_recon).item())

    _, axs = plt.subplots(3, 6)
    axs[0, 0].imshow((image_input_show * 127.5 + 127.5).astype(np.uint8))
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Input')
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')
    for i in range(15):
        if i < 5:
            axs[0, i+1].imshow((images_recon_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[0, i+1].axis('off')
            axs[0, i+1].set_title(f'RMSE {int((i+1)*100)}: {rmse[i]:.3f}')
        elif i < 10:
            axs[1, i-4].imshow((images_recon_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[1, i-4].axis('off')
            axs[1, i-4].set_title(f'RMSE {int((i+1)*100)}: {rmse[i]:.3f}')
        else:
            axs[2, i-9].imshow((images_recon_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[2, i-9].axis('off')
            axs[2, i-9].set_title(f'RMSE {int((i+1)*100)}: {rmse[i]:.3f}')
    plt.show()


if show_low_rank:
    pth_name = pth_path + '1500.pth'
    model.load_state_dict(torch.load(pth_name))
    ranks = np.zeros(12)
    approx_errors = np.zeros(12)
    images_recon_low_rank_show = np.zeros((12, im_shape, im_shape, channel_input))
    for _, (X, _, _) in enumerate(train_loader):
        image_input_show = X[0].permute(1, 2, 0).cpu().detach().numpy()

        _, P = model(X.to(device))
        s = [int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0])), P.shape[1], P.shape[2]]
        print(f'Original rank: {torch.mean(torch.linalg.matrix_rank(P).float()).item():.5f}')
        for r in range(10, 121, 10):
            U, S, Vh = torch.svd_lowrank(P, q=r)
            P_ = U @ torch.diag_embed(S) @ torch.transpose(Vh, 1, 2)
            z_ = torch.reshape(P_, s).permute(3, 2, 0, 1)
            X_recon_low_rank = decoder(z_)
            
            ranks[int(r/10)-1] = torch.mean(torch.linalg.matrix_rank(P_).float()).item()
            approx_errors[int(r/10)-1] = np.sqrt(nn.functional.mse_loss(P_, P).item())
            images_recon_low_rank_show[int(r/10)-1] = np.clip(X_recon_low_rank[0].permute(1, 2, 0).cpu().detach().numpy(), -1., 1.)

    _, axs = plt.subplots(3, 5)
    axs[0, 0].imshow((image_input_show * 127.5 + 127.5).astype(np.uint8))
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Input')
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')
    for i in range(12):
        if i < 4:
            axs[0, i+1].imshow((images_recon_low_rank_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[0, i+1].axis('off')
            axs[0, i+1].set_title(f'Rank: {ranks[i]:.3f}, Approx Error: {approx_errors[i]:.3f}')
        elif i < 8:
            axs[1, i-3].imshow((images_recon_low_rank_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[1, i-3].axis('off')
            axs[1, i-3].set_title(f'Rank: {ranks[i]:.3f}, Approx Error: {approx_errors[i]:.3f}')
        else:
            axs[2, i-7].imshow((images_recon_low_rank_show[i] * 127.5 + 127.5).astype(np.uint8)) 
            axs[2, i-7].axis('off')
            axs[2, i-7].set_title(f'Rank: {ranks[i]:.3f}, Approx Error: {approx_errors[i]:.3f}')
    plt.show()


## Latent space low-rank processing
if process_save_low_rank:
    pth_name = pth_path + '1500.pth'
    model.load_state_dict(torch.load(pth_name))
    recon_low_rank_images = np.zeros((train_size, im_shape, im_shape, channel_input))
    for i, (X, _, _) in enumerate(train_loader):
        _, P = model(X.to(device))
        s = [int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0])), P.shape[1], P.shape[2]]

        U, S, Vh = torch.svd_lowrank(P, q=low_rank)
        P_ = U @ torch.diag_embed(S) @ torch.transpose(Vh, 1, 2)
        z_ = torch.reshape(P_, s).permute(3, 2, 0, 1)

        X_recon_low_rank = decoder(z_)
        X_recon_low_rank = X_recon_low_rank.permute(0, 2, 3, 1).cpu().detach().numpy()
        X_recon_low_rank_image = (X_recon_low_rank * 127.5 + 127.5).astype(np.uint8)

        if X.shape[0] == batch_size:
            recon_low_rank_images[i*batch_size:(i+1)*batch_size] = X_recon_low_rank_image
        else:
            recon_low_rank_images[i*batch_size:] = X_recon_low_rank_image

    for i in range(train_size):
        matplotlib.image.imsave('simulation\\' + sim_data + '\\low_rank_' + str(low_rank) + '\\' + str(i) + '.png', recon_low_rank_images[i].astype(np.uint8))
        if i == train_size - 1:
            print('Finished!')


## Adding random sparse anomalies: Blobs & Lines
if process_save_anomaly:
    recon_low_rank_images = np.zeros((train_size, im_shape, im_shape, channel_input), np.uint8)
    for i in range(train_size):
        img = np.asarray(Image.open('simulation\\' + sim_data + '\\low_rank_' + str(low_rank) + '\\' + str(i) + '.png'))
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        recon_low_rank_images[i] = img
    avg_value = np.squeeze(np.apply_over_axes(np.mean, recon_low_rank_images, (0, 1, 2))).astype(np.uint8)
    
    S_ground_truth = np.zeros((anomaly_number, im_shape, im_shape, channel_input), np.uint8)
    L_background = np.zeros((anomaly_number, im_shape, im_shape, channel_input), np.uint8)
    X_anomaly = np.zeros((anomaly_number, im_shape, im_shape, channel_input), np.uint8)
    
    anomaly_index = random.sample(range(0, train_size-1), anomaly_number)
    for i, idx in enumerate(anomaly_index):
        seedval = np.random.randint(1000)
        if i % 2 == 0:  # Random blobs
            blob = creat_random_blobs(seedval, 200, im_shape, im_shape).astype(np.uint8)
            while blob.sum() == 0:
                seedval = np.random.randint(1000)
                blob = creat_random_blobs(seedval, 200, im_shape, im_shape).astype(np.uint8)
            S_ground_truth[i] = blob
            L_background[i] = recon_low_rank_images[idx].astype(np.uint8)
            X_defect = cv2.add(L_background[i], S_ground_truth[i])
            for j in range(X_defect.shape[2]):
                x = X_defect[:, :, j]
                x[x == 255] = avg_value[j] + color_offset
                X_defect[:, :, j] = x
            X_anomaly[i] = X_defect
        else:           # Random line
            line_thickness = 3
            x1, y1 = np.random.randint(im_shape), np.random.randint(im_shape)
            x2, y2 = np.random.randint(im_shape), np.random.randint(im_shape)
            cv2.line(S_ground_truth[i], (x1, y1), (x2, y2), (255, 255, 255), thickness=line_thickness)
            L_background[i] = recon_low_rank_images[idx].astype(np.uint8)
            X_defect = cv2.add(L_background[i], S_ground_truth[i])
            for j in range(X_defect.shape[2]):
                x = X_defect[:, :, j]
                x[x == 255] = avg_value[j] + color_offset
                X_defect[:, :, j] = x
            X_anomaly[i] = X_defect

    ## Create all folders first
    contam_folder = 'contamination_level_' + str(contamination_level)
    folder_name = 'anomaly1'
    sim_folder = 'simulation\\' + sim_data + '\\' + contam_folder + '\\' + folder_name
    os.makedirs(sim_folder, exist_ok=True)

    out_root = os.path.join('data', 'btad_simulation',
                            '02_sim_contam_level_' + str(contamination_level), 'train')
    os.makedirs(os.path.join(out_root, 'good'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'defect'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'defect_background'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'defect_ground_truth'), exist_ok=True)

    ## Summary figure
    _, axs = plt.subplots(int(np.ceil(anomaly_number/2)), 6)
    for i in range(anomaly_number):
        if i % 2 == 0:
            axs[int(i/2), 0].imshow(L_background[i])
            axs[int(i/2), 0].axis('off')
            axs[int(i/2), 0].set_title(f'Background L: #{anomaly_index[i]}')
            axs[int(i/2), 1].imshow(S_ground_truth[i])
            axs[int(i/2), 1].axis('off')
            axs[int(i/2), 1].set_title(f'Ground Truth S: #{anomaly_index[i]}')
            axs[int(i/2), 2].imshow(X_anomaly[i])
            axs[int(i/2), 2].axis('off')
            axs[int(i/2), 2].set_title(f'Anomalous X: #{anomaly_index[i]}')
        else:
            axs[math.floor(i/2), 3].imshow(L_background[i])
            axs[math.floor(i/2), 3].axis('off')
            axs[math.floor(i/2), 3].set_title(f'Background L: #{anomaly_index[i]}')
            axs[math.floor(i/2), 4].imshow(S_ground_truth[i])
            axs[math.floor(i/2), 4].axis('off')
            axs[math.floor(i/2), 4].set_title(f'Ground Truth S: #{anomaly_index[i]}')
            axs[math.floor(i/2), 5].imshow(X_anomaly[i])
            axs[math.floor(i/2), 5].axis('off')
            axs[math.floor(i/2), 5].set_title(f'Anomalous X: #{anomaly_index[i]}')
    # plt.savefig(sim_folder + '\\simulation_' + str(contamination_level) + '.png', dpi=150, bbox_inches='tight')
    plt.show()

    ## Save intermediate anomaly files
    for i in range(anomaly_number):
        matplotlib.image.imsave(sim_folder + '\\S_' + str(anomaly_index[i]) + '.png', S_ground_truth[i])
        matplotlib.image.imsave(sim_folder + '\\L_' + str(anomaly_index[i]) + '.png', L_background[i])
        matplotlib.image.imsave(sim_folder + '\\X_' + str(anomaly_index[i]) + '.png', X_anomaly[i])

    # Save mixed dataset to "good": normal images with defective samples substituted at their indices
    anomaly_map = {anomaly_index[j]: j for j in range(anomaly_number)}
    for i in range(train_size):
        if i in anomaly_map:
            img_to_save = X_anomaly[anomaly_map[i]]
        else:
            img_to_save = recon_low_rank_images[i]
        matplotlib.image.imsave(os.path.join(out_root, 'good', str(i) + '.png'),
                                img_to_save.astype(np.uint8))

    # Save anomalous samples indexed by source index
    for i in range(anomaly_number):
        idx = anomaly_index[i]
        matplotlib.image.imsave(os.path.join(out_root, 'defect', str(idx) + '.png'),
                                X_anomaly[i].astype(np.uint8))
        matplotlib.image.imsave(os.path.join(out_root, 'defect_background', str(idx) + '.png'),
                                L_background[i].astype(np.uint8))
        # Ground truth: single-channel binary mask (take first channel, already 0/255)
        matplotlib.image.imsave(os.path.join(out_root, 'defect_ground_truth', str(idx) + '.png'),
                                S_ground_truth[i, :, :, 0].astype(np.uint8), cmap='gray')
    print(f'Saved simulation data to {out_root} (contamination_level={contamination_level})')


