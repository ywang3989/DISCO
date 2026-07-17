import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_dme_result(x_aixs, metrics, f):
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(9, 7))
    plt.plot(x_aixs, metrics.transpose(), label=['RPCA: ' + str(np.round(np.trapz(metrics[0, :], dx=x_aixs[1]-x_aixs[0]), 3)), 
                                                 'RDAE: ' + str(np.round(np.trapz(metrics[1, :], dx=x_aixs[1]-x_aixs[0]), 3)), 
                                                 'MemAE: ' + str(np.round(np.trapz(metrics[2, :], dx=x_aixs[1]-x_aixs[0]), 3)), 
                                                 'DRAEM: ' + str(np.round(np.trapz(metrics[3, :], dx=x_aixs[1]-x_aixs[0]), 3)), 
                                                 'DISCO-wo-EP: ' + str(np.round(np.trapz(metrics[4, :], dx=x_aixs[1]-x_aixs[0]), 3)), 
                                                 'DISCO-wo-P: ' + str(np.round(np.trapz(metrics[5, :], dx=x_aixs[1]-x_aixs[0]), 3)),
                                                 'DISCO-wo-E: ' + str(np.round(np.trapz(metrics[6, :], dx=x_aixs[1]-x_aixs[0]), 3)),
                                                 'DISCO: ' + str(np.round(np.trapz(metrics[7, :], dx=x_aixs[1]-x_aixs[0]), 3))])
    if f == 'psnr':
        plt.ylim((0, 40))
        plt.ylabel(r'$DMS_{5}$-$PSNR$')
        plt.title(r'Visulization of $DMS_{5}$-$PSNR$')
    elif f == 'ssim':
        plt.ylim((0, 1))
        plt.ylabel(r'$DMS_{5}$-$SSIM$')
        plt.title(r'Visulization of $DMS_{5}$-$SSIM$')
    plt.xlabel('Interpolation')
    plt.legend(ncol=2)
    plt.show()


data = pd.read_excel('temp_var\\DMS5.xlsx')
data = data.to_numpy()

dms_line = data[0:16, :]
dms_psnrs_line = dms_line[0::2, 2:].astype(np.float32)
dms_ssims_line = dms_line[1::2, 2:].astype(np.float32)
dms_blob = data[17:, :]
dms_psnrs_blob = dms_blob[0::2, 2:].astype(np.float32)
dms_ssims_blob = dms_blob[1::2, 2:].astype(np.float32)

K = 5
x_aixs = np.linspace(0, 1, num=K+1)
plot_dme_result(x_aixs, dms_ssims_blob, 'ssim')



