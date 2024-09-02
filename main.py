import os, sys
from torchstat import stat
import random
import time
from datetime import timedelta
from config import get_config
from utils.SARop import CSA_echo, CSA_imag
import utils.train_torch as train
from utils.train_torch import VideoDataset
from scipy.io import loadmat, savemat
import numpy as np
import torch
from torch import nn
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from utils.train_torch import do_training
#try :
#    from PIL import Image
#    from sklearn.feature_extraction.image \
#            import extract_patches_2d, reconstruct_from_patches_2d
#except Exception as e :
#    pass

def setup_model (config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf  = 'c' if config.coord  else 's'

    """ViSARUTNet"""
    config.model = ("ViSARUTNet_T{T}_M{M}_N{N}_D{depth}_Samp{sample}"
                    .format (T=config.T, M=config.M,
                            N=config.N, depth=config.D, sample=config.sample_rate))
    from models.ViSARUTNet import ViSARUTNet_Net
    model = ViSARUTNet_Net(config, mask=config.mask, thetas=config.thetas).cuda()

    config.modelfn = os.path.join (config.expbase, config.model)
    config.resfn   = os.path.join (config.resbase, config.model)
    print ("model disc:", config.model)
    return model

def run_sc_train(config):
    # start timer
    model = setup_model(config)
    train_data = VideoDataset('data/train_data')
    valid_data = VideoDataset('data/val_data')
    start = time.time()
    do_training(config, model, train_data, valid_data)
    end = time.time()
    elapsed = end - start
    print("elapsed time of training = " + str(timedelta(seconds=elapsed)))


############################################################
######################   Testing    ########################
############################################################
def run_sc_test(config):
    """
    Test model.
    """

    #train_data = VideoDataset('data/train_data15')
    """Set up model."""
    if config.net == 'LRS':
        model = setup_model(config, invr=10, ldr=0.3, rho=0.1)
    else:
        model = setup_model(config)
    #print(model)
    """Create session and initialize the graph."""
    # load model
    model.load_trainable_variables(config.modelfn)

    model.requires_grad_(False)
    #plt.plot(model.state['nmse_hist_val'])
    test_data = loadmat("data/test_data/1_56.mat")
    data = torch.tensor(test_data['data'], dtype=torch.float32)
    test_data = data.unsqueeze(0).unsqueeze(0)
    test_x = test_data.cuda()
    y = CSA_echo(test_x, config.thetas)
    std = (torch.var(y, dim=[3, 4], keepdim=True).sqrt()
           * torch.tensor(np.power(10.0, -config.SNR / 20.0), dtype=torch.float32))
    noise = torch.randn_like(y) * std
    y = (y + noise) * config.mask  # 加入噪声并进行降采样


    X = model.inference(y)
    fram_id = 0
    X = X[:, :, fram_id, :, :]
    X = torch.abs(X)
    X = nn.functional.normalize(X.reshape(-1, config.M * config.N),
                                float('inf')).reshape(-1, 1, config.M, config.N)
    #plt.figure(), plt.imshow(X[0,0,6,:,:].cpu(), cmap='gray')
    x_nni = X[0, 0, :, :].cpu().numpy()
    x_lab=test_x[0,0,fram_id,:, :].cpu().numpy()
    mse_nni = np.mean((x_lab - x_nni) ** 2)  # 计算均方误差（MSE）
    max_valuenni = np.max(x_nni)  # 张量1的动态范围
    rmse_nni = np.sqrt(mse_nni)
    psnr_nni = 20 * np.log10(max_valuenni / rmse_nni)  # 计算PSNR
    ssim_nni = structural_similarity(x_lab, x_nni, gaussian_weights=True)
    print('rmse', 'psnr', 'ssim')
    print(rmse_nni,psnr_nni,ssim_nni)
    plt.figure(), plt.imshow(x_nni, cmap='gray')
    plt.figure(), plt.imshow(x_lab, cmap='gray')
    plt.show()

def main ():
    #assert os.path.dirname(__file__).lower().replace('\\','/') == os.getcwd().lower().replace('\\','/'), \
     #   'Please run under the directory of the .py file!'
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    torch.cuda.set_device(int(config.gpu))
    mask = loadmat('mask_'+str(config.sample_rate)+'.mat')  # 严谨起见可加文件是否存在的检测
    config.mask = torch.cuda.BoolTensor(mask['mask'])
    print('Mask loaded successfully!')
    thetas = loadmat('thetas.mat')
    Theta1 = torch.tensor(thetas['Theta1'], dtype=torch.complex64).cuda()
    Theta2 = torch.tensor(thetas['Theta2'], dtype=torch.complex64).cuda()
    Theta3 = torch.tensor(thetas['Theta3'], dtype=torch.complex64).cuda()
    config.thetas = [Theta1, Theta2, Theta3]
    print('Thetas loaded successfully!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #if config.test:
    #run_test (config)
    #else:

    if config.test:
        run_sc_test(config)
    else:
        run_sc_train(config)
    # end of main

if __name__ == "__main__":
    main ()