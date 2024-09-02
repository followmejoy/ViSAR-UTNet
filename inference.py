from __future__ import print_function
import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import time
import scipy.io
import numpy as np

from utils.models import Transformer
from torch.utils.data import DataLoader
from torchvision import transforms
from load_data import TrainDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils.SARop import CSA_echo

test_data_path = 'data/test_data'
checkpoint_path = 'checkpoints/ViSARUTNet.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 'cpu' or 'cuda'
mask_id = 20
mask = loadmat('mask_{}.mat'.format(mask_id))  # 严谨起见可加文件是否存在的检测

############ global variables ##########################3
size_batch = 1  # minibatch size
num_batch = 1  # number of batches

# --------------- 读入数据---------------------------------------------
transform = transforms.Compose(transforms.ToTensor())
testdata = TrainDataset(test_data_path, transform)
testdata_loader = DataLoader(testdata, batch_size=size_batch, shuffle=False)

mask = torch.tensor(mask['mask'], dtype=torch.bool).to(device)
print('Mask loaded successfully!')
# --------------- 加载模型 ---------------------------------------------
checkpoint = torch.load(checkpoint_path, map_location=device)
model_opt = checkpoint['opt']

model = Transformer(model_opt)
model = model.to(device)

model.load_state_dict(checkpoint['model_params'])
print('Loaded pre-trained model_state..')

criterion = nn.MSELoss()
for i, X in enumerate(testdata_loader):
    t_1 = time.time()
    X = X.float().to(device)
    Y = CSA_echo(X, model_opt.thetas)
    Y = Y.squeeze(1)
    X = X.squeeze(1)
    """Add noise with SNR."""
    std = (torch.var(Y, dim=[1, 2], keepdim=True).sqrt()
           * torch.tensor(np.power(10.0, -model_opt.SNR / 20.0), dtype=torch.float32))
    noise = torch.randn_like(Y) * std
    Y = (Y + noise) * mask  # 加入噪声并进行降采样
    Y = torch.cat((Y.real, Y.imag), dim=2)
    X = torch.cat((X, torch.zeros_like(X)), dim=2)
    dec_logits, _, _, _ = model(Y, X)
    t_2 = time.time()
    step_loss = criterion(dec_logits, X.contiguous())
    print('test-loss-total:{}, time:{}'.format(step_loss, t_2-t_1))

    x_result = dec_logits.float().squeeze(0).detach().cpu().numpy()
    y = Y.squeeze(0).detach().cpu().numpy()
    x_label = X.squeeze(0).detach().cpu().numpy()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Transformer')
    plt.imshow(x_result)
    plt.subplot(1, 2, 2)
    plt.imshow(x_label)
    plt.show()

    # result_patch = 'results/result_ViSARUTNet_d1_mask{}_m{}_snr{}_{}.mat'.format(mask_id, model_opt.d_model,
    #                                                                            model_opt.SNR, i + 1)
    # scipy.io.savemat(result_patch, {'x_result': x_result, 'echo': y, 'x_label': x_label,
    # 'time': t_2 - t_1, 'loss': step_loss})

print('# Terminated #')
