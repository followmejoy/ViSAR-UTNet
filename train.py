from __future__ import print_function
import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
from scipy.io import loadmat
import numpy as np

from utils.models import Transformer
from utils.optimizer import ScheduledOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from load_data import TrainDataset
from utils.SARop import CSA_echo

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 'cpu' or 'cuda'


def create_model(opt, checkpoint_path=None):
# If opt.model_path exists, load model parameters.
# load_train_model = True
    if checkpoint_path:
        print('Loading half-trained model parameters..')
        # load an existing one.
        model_state = torch.load(checkpoint_path, map_location=device)
        model_opt = model_state['opt']
        model = Transformer(model_opt)
        model.load_state_dict(model_state['model_params'])
    else:
        print('Creating new model parameters..')
        model = Transformer(opt)  # Initialize a model state.
        model_state = {'opt': opt, 'curr_epochs': 0, 'train_steps': 0}

    if use_cuda:
        print('Using GPU..')
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select the GPU device
        # model = model.cuda()
        model = model.to(device)

    return model, model_state


def main(opt):
    print('Loading training and development data..')

    transform = transforms.Compose(transforms.ToTensor())
    traindata = TrainDataset(opt.train_data_path, transform)
    evaldata = TrainDataset(opt.eval_data_path, transform)
    traindata_loader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=True)
    eval_loader = DataLoader(evaldata, batch_size=opt.batch_size, shuffle=True)

    # Create a new model or load an existing one.
    model, model_state = create_model(opt, 'checkpoints/ViSARUTNet_d1_m1024_snrinf_samp70_epoch100.pth')
    init_epoch = model_state['curr_epochs']
    if init_epoch >= opt.max_epochs:
        print('Training is already complete.',
              'current_epoch:{}, max_epoch:{}'.format(init_epoch, opt.max_epochs))
        sys.exit(0)

    # Loss and Optimizer
    # If size_average=True (default): Loss for a mini-batch is averaged over non-ignore index targets.
    # criterion = nn.CrossEntropyLoss(size_average=False)
    criterion = nn.MSELoss()
    optimizer = ScheduledOptimizer(optim.Adam(model.trainable_params(), betas=(0.9, 0.98), eps=1e-9),
                                   opt.d_model, opt.n_layers, opt.n_warmup_steps)
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_dev_file = opt.log + '.valid.log'
        if not os.path.exists(log_train_file) and os.path.exists(log_dev_file):
            with open(log_train_file, 'w') as log_tf, open(log_dev_file, 'w') as log_df:
                log_tf.write('epoch,ppl,sents_seen\n')
                log_df.write('epoch,ppl,sents_seen\n')
        print('Training and validation log will be written in {} and {}'
              .format(log_train_file, log_dev_file))

    for epoch in range(init_epoch + 1, opt.max_epochs + 1):
        # Execute training steps for 1 epoch.
        train_loss = train(model, criterion, optimizer, traindata_loader, model_state)
        print('Epoch {}'.format(epoch), 'Train_ppl: {0:.2f}'.format(train_loss))

        # Execute a validation step.
        eval_loss = eval(model, opt, criterion, eval_loader)
        print('Epoch {}'.format(epoch), 'Eval_ppl: {0:.2f}'.format(eval_loss))

        if epoch % opt.save_freq == 0:
             # Save the model checkpoint in every 1 epoch.
            model_state['curr_epochs'] = epoch
            model_state['model_params'] = model.state_dict()

            torch.save(model_state, 'checkpoints/ViSARUTNet_d1_m{}_snr{}_samp{}_epoch{}.pth'.format(opt.d_model, opt.SNR, opt.sample_rate, epoch))
            print('### The model checkpoint file has been saved!')

        if opt.log and log_train_file and log_dev_file:
            with open(log_train_file, 'a') as log_tf, open(log_dev_file, 'a') as log_df:
                log_tf.write('{epoch},{ppl:0.2f},{sents}\n'.format(
                    epoch=epoch, ppl=train_loss))
                log_df.write('{epoch},{ppl:0.2f},{sents}\n'.format(
                    epoch=epoch, ppl=eval_loss))


def train(model, criterion, optimizer, train_iter, model_state):  # TODO: fix opt
    model.train()
    opt = model_state['opt']
    train_loss_total = 0.0

    start_time = time.time()

    for i, X in enumerate(train_iter):

        # Execute a single training step: forward
        optimizer.zero_grad()
        X = X.float().to(device)
        Y = CSA_echo(X, opt.thetas)
        Y = Y.squeeze(1)
        X = X.squeeze(1)
        """Add noise with SNR."""
        std = (torch.var(Y, dim=[1, 2], keepdim=True).sqrt()
               * torch.tensor(np.power(10.0, -opt.SNR / 20.0), dtype=torch.float32))
        noise = torch.randn_like(Y) * std
        Y = (Y + noise) * opt.mask 
        Y = torch.cat((Y.real, Y.imag), dim=2)
        X = torch.cat((X, torch.zeros_like(X)), dim=2)
        dec_logits, _, _, _ = model(Y, X)
        step_loss = criterion(dec_logits, X.contiguous())

        # Execute a single training step: backward
        step_loss.backward()
        if opt.max_grad_norm:
            clip_grad_norm(model.trainable_params(), float(opt.max_grad_norm))
        optimizer.step()
        optimizer.update_lr()
        model.proj_grad()  

        train_loss_total += float(step_loss.item())
        model_state['train_steps'] += 1

        # Display training status
        if model_state['train_steps'] % opt.display_freq == 0:
            time_elapsed = (time.time() - start_time)
            step_time = time_elapsed / opt.display_freq

            print('Epoch {0:<3}'.format(model_state['curr_epochs']), 'Step {0:<10}'.format(model_state['train_steps']),
                  'Step-time {0:<10.2f}'.format(step_time), 'Train-loss-total {0:<10.2f}'.format(train_loss_total))
            start_time = time.time()

    # return per_word_loss over 1 epoch
    return train_loss_total


def eval(model, opt, criterion, dev_iter):
    model.eval()
    eval_loss_total = 0.0

    print('Evaluation')
    with torch.no_grad():
        for i, X in enumerate(dev_iter):
            X = X.float().to(device)
            Y = CSA_echo(X, opt.thetas)
            Y = Y.squeeze(1)
            X = X.squeeze(1)
            """Add noise with SNR."""
            std = (torch.var(Y, dim=[1, 2], keepdim=True).sqrt()
                   * torch.tensor(np.power(10.0, -opt.SNR / 20.0), dtype=torch.float32))
            noise = torch.randn_like(Y) * std
            Y = (Y + noise) * opt.mask  
            Y = torch.cat((Y.real, Y.imag), dim=2)
            X = torch.cat((X, torch.zeros_like(X)), dim=2)
            dec_logits, _, _, _ = model(Y, X)
            step_loss = criterion(dec_logits, X.contiguous())
            eval_loss_total += float(step_loss.item())
        print('eval-loss-ave:{}'.format(eval_loss_total/len(dev_iter)))

    # return per_word_loss
    return eval_loss_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparams')
    # data loading params
    # parser.add_argument('-data_path', required=True, help='Path to the preprocessed data', default='data/TrainData')
    parser.add_argument('-train_data_path', type=str, default='data/train_data1')
    parser.add_argument('-eval_data_path', type=str, default='data/val_data1')
    # parser.add_argument('-data_path', type=str, default='data/TrainData')

    # network params
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=128)  #64
    parser.add_argument('-d_v', type=int, default=128)  #64
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.01)
    parser.add_argument('-share_proj_weight', action='store_true')
    parser.add_argument('-share_embs_weight', action='store_true')
    parser.add_argument('-weighted_model', action='store_true')

    # training params
    parser.add_argument('-sample_rate', type=int, default=70)
    parser.add_argument('-SNR', type=float, default=float('inf'))
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-max_epochs', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=300)
    parser.add_argument('-save_freq', type=int, default=10)
    parser.add_argument('-log', default=None)
    parser.add_argument('-model_path', type=str, default='checkpoints')

    opt = parser.parse_args()
    mask = loadmat('mask_' + str(opt.sample_rate) + '.mat')  
    opt.mask = torch.cuda.BoolTensor(mask['mask'])
    print('Mask loaded successfully!')
    thetas = loadmat('thetas.mat')
    Theta1 = torch.tensor(thetas['Theta1'], dtype=torch.complex64).cuda()
    Theta2 = torch.tensor(thetas['Theta2'], dtype=torch.complex64).cuda()
    Theta3 = torch.tensor(thetas['Theta3'], dtype=torch.complex64).cuda()
    opt.thetas = [Theta1, Theta2, Theta3]
    print('Thetas loaded successfully!')
    # print(opt)

    main(opt)
    print('Terminated')
