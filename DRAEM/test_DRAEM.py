import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
import skimage

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred, data_range=255):
    '''For 8-bit images'''
    return 10 * np.log10(data_range**2 / np.mean((y_true - y_pred)**2))


def structural_similarity(y_true, y_pred, data_range=255):
    '''For 8-bit images'''
    y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
    mu_x, sigma_x_sq = np.mean(y_true_flat), np.var(y_true_flat, ddof=1)
    mu_y, sigma_y_sq = np.mean(y_pred_flat), np.var(y_pred_flat, ddof=1)
    sigma_xy = (1/(np.size(y_true_flat)-1)) * np.dot(y_true_flat-mu_x, y_pred_flat-mu_y)
    c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
    A1 = 2 * mu_x * mu_y + c1
    A2 = 2 * sigma_xy + c2
    B1 = mu_x**2 + mu_y**2 + c1
    B2 = sigma_x_sq + sigma_y_sq + c2
    D = B1 * B2
    return (A1 * A2) / D


def find_minimum_bounding_box(M):
    '''M: {0,1}^(H x W) - 2D binary mask'''
    indices = np.where(M == 1.)
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    return [y_min, y_max, x_min, x_max]


def generate_bounding_boxes(mask, interpolation_points):
    '''interplation_points: [0, ..., 1] - 1D'''
    img_size = mask.shape[0]
    min_box_ind = find_minimum_bounding_box(mask)
    y_min, y_max = min_box_ind[0], min_box_ind[1]
    x_min, x_max = min_box_ind[2], min_box_ind[3]

    top_left_y = [y_min, 0]  # top_right_y
    top_left_x = [x_min, 0]  # bottom_left_x
    top_right_x = [x_max, img_size]  # bottom_right_x
    bottom_left_y = [y_max, img_size]  # bottom_right_y

    new_points = interpolation_points[1:-1]
    top_left_y_new = np.interp(new_points, [0, 1], top_left_y).astype(np.int8)
    top_left_x_new = np.interp(new_points, [0, 1], top_left_x).astype(np.int8)
    top_right_x_new = np.interp(new_points, [0, 1], top_right_x).astype(np.int8)
    bottom_left_y_new = np.interp(new_points, [0, 1], bottom_left_y).astype(np.int8)

    return {'top left & right y': np.concatenate((np.array([y_min]), top_left_y_new, np.array([0]))),
            'top & bottom left x': np.concatenate((np.array([x_min]), top_left_x_new, np.array([0]))),
            'top & bottom right x': np.concatenate((np.array([x_max]), top_right_x_new, np.array([img_size-1]))),
            'bottom left & right y': np.concatenate((np.array([y_max]), bottom_left_y_new, np.array([img_size-1])))}


def DMS_f(mask, interpolation_points, func, Y_true, Y_pred, data_range=255):
    boxes_ind = generate_bounding_boxes(mask, interpolation_points)

    metrics = np.zeros(interpolation_points.shape)
    for i in range(interpolation_points.shape[0]):
        y_min, y_max = boxes_ind['top left & right y'][i], boxes_ind['bottom left & right y'][i]
        x_min, x_max = boxes_ind['top & bottom left x'][i], boxes_ind['top & bottom right x'][i]
        y_true = Y_true[y_min:y_max+1, x_min:x_max+1, :]
        y_pred = Y_pred[y_min:y_max+1, x_min:x_max+1, :]
        metrics[i] = func(y_true, y_pred, data_range)

    dms_f = np.trapz(metrics, dx=interpolation_points[1]-interpolation_points[0])
    return metrics, dms_f


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    for obj_name in obj_names:
        img_dim = 128
        run_name = base_model_name + "_" + obj_name + '_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        whether_plot = False  # False to choos best threshold
        thre_best = 0.05  # 02_sim1: 0.06, 02_sim2 & 02_sim3 & contam_level_0.05: 0.05, BTAD3_Sys: 0.66
        thre_upper = 2
        threholds = np.linspace(0.01, thre_upper, int(100*thre_upper))
        metrics_thre = np.zeros((int(100*thre_upper), 8))
        for idx, thre in enumerate(threholds):
            dice_coefs = []
            auroc_vals = []
            bkgd_psnrs = []
            bkgd_ssims = []

            for i, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                gray_rec = model(gray_batch)

                ## Evaluation
                img_path = sample_batched["img_path"]
                img_idx = os.path.basename(img_path[0])

                S_defect = gray_batch - gray_rec
                S_defect_show = np.sum(np.abs(S_defect.squeeze().permute(1, 2, 0).cpu().detach().numpy()), axis=2)

                X_image_show = (gray_batch.squeeze().permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
                X_image_show = cv2.cvtColor(X_image_show, cv2.COLOR_BGR2RGB)

                L_backgd_show = (np.clip(gray_rec.squeeze().permute(1, 2, 0).cpu().detach().numpy(), 0., 1.) * 255.0).astype(np.uint8)
                L_backgd_show = cv2.cvtColor(L_backgd_show, cv2.COLOR_BGR2RGB)

                true_mask = sample_batched["mask"]
                S_ground_truth = true_mask.squeeze().cpu().detach().numpy()
                S_ground_truth[S_ground_truth != 0] = 1

                # Anomaly detection
                if whether_plot:
                    S_mask_pred = (S_defect_show > thre_best).astype(int)
                else:
                    S_mask_pred = (S_defect_show > thre).astype(int)
                dice_coef = dice_coefficient(S_ground_truth, S_mask_pred)
                auroc_val = roc_auc_score(S_ground_truth.flatten(), S_mask_pred.flatten())
                dice_coefs.append(dice_coef)
                auroc_vals.append(auroc_val)

                # Background restoration
                if obj_name in ['02_sim1', '02_sim2', '02_sim3', '02_sim_contam_level_0.05', '02_sim_contam_level_0.08', '02_sim_contam_level_0.1']:
                    L_ground_truth_show = mpimg.imread('./datasets/data/' + obj_name + '/train/defect_background/' + img_idx)
                else:
                    L_ground_truth_show = np.zeros((img_dim, img_dim, 3))
                if L_ground_truth_show.shape[2] == 4:
                    L_ground_truth_show = L_ground_truth_show[:, :, 0:3]

                K = 20
                interpol_pts = np.linspace(0, 1, num=K+1)
                L_ground_truth_ = (L_ground_truth_show * 255).astype(np.int16)
                L_backgd_ = L_backgd_show.astype(np.int16)
                psnrs, dms_psnr = DMS_f(S_ground_truth, interpol_pts, peak_signal_to_noise_ratio, L_ground_truth_, L_backgd_)
                ssims, dms_ssim = DMS_f(S_ground_truth, interpol_pts, structural_similarity, L_ground_truth_, L_backgd_)
                bkgd_psnrs.append(dms_psnr)
                bkgd_ssims.append(dms_ssim)

                # print('---------------------')
                # print('DMS_' + str(K) + '-PSNR: ', psnrs)
                # print('DMS_' + str(K) + '-SSIM: ', ssims)
                # np.savetxt('dms_draem_02_sim2_' + str(i+1) + '.csv', np.concatenate((np.expand_dims(psnrs, 0), np.expand_dims(ssims, 0)), 0), delimiter=',')
                
                if whether_plot:
                    _, axs = plt.subplots(2, 3)
                    axs[0, 0].imshow(X_image_show)
                    axs[0, 0].axis('off')
                    axs[0, 0].set_title('Input X')
                    axs[0, 1].imshow(L_ground_truth_show)
                    axs[0, 1].axis('off')
                    axs[0, 1].set_title('Background L')
                    axs[0, 2].imshow(L_backgd_show) 
                    axs[0, 2].axis('off')
                    axs[0, 2].set_title('Restored L\nPSNR: ' + str(round(dms_psnr, 3)) + ' & SSIM: ' + str(round(dms_ssim, 3)))
                    with plt.style.context('classic'):
                        axs[1, 0].imshow(S_defect_show)
                        axs[1, 0].axis('off')
                        axs[1, 0].set_title('Anomaly Map |S|')
                        norm1 = colors.Normalize(0, np.max(S_defect_show))
                        plt.colorbar(cm.ScalarMappable(norm=norm1), ax=axs[1, 0])
                    with plt.style.context('grayscale'):
                        axs[1, 1].imshow(S_ground_truth)
                        axs[1, 1].axis('off')
                        axs[1, 1].set_title('True Mask A')
                        axs[1, 2].imshow(S_mask_pred)
                        axs[1, 2].axis('off')
                        axs[1, 2].set_title('Pred. Mask |S|>t\nDice: ' + str(round(dice_coef, 3)) + ' & AUROC: ' + str(round(auroc_val, 3)))
                        norm2 = colors.Normalize(0, 1)
                        plt.colorbar(cm.ScalarMappable(norm=norm2), ax=axs[1, 2])
                    plt.show()

            if whether_plot:
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
                                                 'PSNR Avg: ' + str(np.round(metrics_thre[max_idx, 4], 3)), 
                                                 'PSNR Std: ' + str(np.round(metrics_thre[max_idx, 5], 3)),
                                                 'SSIM Avg: ' + str(np.round(metrics_thre[max_idx, 6], 3)), 
                                                 'SSIM Std: ' + str(np.round(metrics_thre[max_idx, 7], 3))])
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
        plt.title(obj_name)
        plt.yscale('log')
        plt.legend()
        plt.show()

        quit()



def save_train_S(obj_names, checkpoint_path, base_model_name, out_dir):
    """Run DRAEM reconstruction on training defect images and save S_defect tensors.

    Saves temp_var/{category}/S_{category}_draem.pt, a tensor of shape
    (max_filename_num + 1, 3, 128, 128) indexed by integer filename number,
    matching the indexing scheme used by MVTEC dataset in roc_pixelevel.py.
    """
    img_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for category in obj_names:
        run_name = base_model_name + '_' + category + '_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + '.pckl'), map_location=device))
        model.to(device)
        model.eval()

        defect_dir = os.path.join('./datasets/data', category, 'train', 'defect')
        files = sorted(f for f in os.listdir(defect_dir) if f.endswith('.png'))

        S_dict = {}
        for fname in files:
            idx = int(os.path.splitext(fname)[0])
            img = cv2.imread(os.path.join(defect_dir, fname), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_dim, img_dim))
            img = img.astype(np.float32) / 255.0
            X_input = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                gray_rec = model(X_input)
                S_defect = X_input - gray_rec  # (1, 3, H, W)

            S_dict[idx] = S_defect.squeeze(0).cpu()

        max_idx = max(S_dict.keys())
        S_tensor = torch.zeros(max_idx + 1, 3, img_dim, img_dim)
        for idx, s in S_dict.items():
            S_tensor[idx] = s

        out_path = os.path.join(out_dir, category, f'S_{category}_draem.pt')
        torch.save(S_tensor, out_path)
        print(f'Saved {out_path}  (shape: {S_tensor.shape}, {len(S_dict)} defect images)')


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    obj_list = ['02_sim1', '02_sim2', '02_sim3', 'BTAD3_sys', 'wood_mix', '02_sim_contam_level_0.05', '02_sim_contam_level_0.08', '02_sim_contam_level_0.1']

    with torch.cuda.device(args.gpu_id):
        print(obj_list[int(args.obj_id)])
        test([obj_list[int(args.obj_id)]], args.data_path, args.checkpoint_path, args.base_model_name)

# if __name__ == "__main__":
#     save_train_S(
#         obj_names=['02_sim_contam_level_0.05', '02_sim_contam_level_0.08', '02_sim_contam_level_0.1'],
#         checkpoint_path='./checkpoints',
#         base_model_name='DRAEM_test_0.0001_500_bs8',
#         out_dir='../Codes/temp_var'
#     )
