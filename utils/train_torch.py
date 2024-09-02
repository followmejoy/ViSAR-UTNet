
from collections import OrderedDict
import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
from utils.SARop import CSA_echo

@torch.no_grad()
def inference_mode():
    pass

class ImageDataset(Dataset):
    def __init__(self,mode='Train', num_per_cat=300):
        urb1 = loadmat('data/urb1_rand.mat')
        siz = urb1['data'].shape[2] 
        if mode == 'Train':
            assert num_per_cat <= 300, 'Please try another num_per_cat <= 300!'
            urb1 = torch.tensor(urb1['data'], dtype=torch.float32)[...,:num_per_cat]
        elif mode == 'Valid':
            assert num_per_cat <= 50, 'Please try another num_per_cat <= 50!'
            urb1 = torch.tensor(urb1['data'], dtype=torch.float32)[...,300:300+num_per_cat]
        elif mode == 'Test':
            assert num_per_cat <= siz-350, f'Please try another num_per_cat <= {siz-350}!'
            urb1 = torch.tensor(urb1['data'], dtype=torch.float32)[...,350:350+num_per_cat]
        else:
            raise ValueError(f'Expected mode: Train, Valid, Test, got {mode}')
        self.data_all = urb1.unsqueeze(0).permute(3,0,1,2) 
    def __len__(self):
        return self.data_all.shape[0]

    def __getitem__(self, idx):
        image = self.data_all[idx, ...]
        return image


class VideoDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.folder_path, file_name)
        data = loadmat(file_path)  

        tensor_data = torch.tensor(data['data'],dtype=torch.float32) 
        tensor_data = tensor_data.unsqueeze(0)
        # tensor_data=tensor_data.unsqueeze(0).permute(0,3,1,2) 

        return tensor_data



#dataset = CustomDataset(folder_path='path/to/folder')
#data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def setup_input_sc (test, tbs, vbs, val_step, its, SNR,
                    train_data, valid_data, mask, thetas):
    """TODO: Docstring for function.
    :arg1: TODO
    :returns: TODO
    """
    train_loader = DataLoader(train_data, batch_size=tbs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=vbs, shuffle=True)
    y_, x_, y_val_, x_val_ = [], [], [], []

    for x in train_loader:
        x = x.cuda()
        y = CSA_echo(x, thetas)
        """Add noise with SNR."""
        std = (torch.var (y, dim=[3, 4], keepdim=True).sqrt()
            * torch.tensor(np.power (10.0, -SNR/20.0), dtype=torch.float32))
        noise = torch.randn_like(y) * std
        y = (y + noise) * mask 
        x_.append(x.cpu())
        y_.append(y.cpu())

    if not test:
        # val_nums = len(train_loader) // val_step 
        # val_nums = val_nums+1 if its%len(train_loader) else val_nums
        val_nums = len(valid_loader)
        for _ in range(val_nums):
            # valid_loader = DataLoader(valid_data, batch_size=vbs, shuffle=True)
            x_val = next(iter(valid_loader))
            x_val = x_val.cuda()
            y_val = CSA_echo(x_val, thetas)
            std_val = (torch.var (y_val, dim=[3, 4], keepdim=True).sqrt()
                * torch.tensor(np.power (10.0, -SNR/20.0), dtype=torch.float32))
            noise_val = torch.randn_like(y_val) * std_val
            y_val = (y_val + noise_val) * mask
            x_val_.append(x_val.cpu())
            y_val_.append(y_val.cpu())

    """In the order of `input_, label_, input_val_, label_val_`."""
    if not test:
        return y_, x_, y_val_, x_val_
    else:
        return y_, x_


def do_training (config, model, train_data, valid_data):
    """
    Train the model actually.

    :sess: Tensorflow session. Variables should be initialized or loaded from trained
           model in this session.
    :stages: Training stages info. ( name, xh_, loss_, nmse_, op_, var_list ).
    :prob: Problem instance.
    :batch_size: Batch size.
    :val_step: How many steps between two validation.
    :maxit: Max number of iterations in each training stage.
    :better_wait: Jump to next training stage in advance if nmse_ no better after
                  certain number of steps.
    :done: name of stages that has been done.

    """
    if config.net == 'LISTA_gen' or 'selfatt_LISTA' or 'LISTA_gen_align':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.init_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    itmax = 0
    for fn in os.listdir(config.expbase):
        sp1 = fn.split(config.model)  #config.model=LISTA_gen_T6_t_s_40
        if len(sp1) < 2:
            continue
        sp2 = sp1[1].split('.pt')[0]
        if sp2 == '_best_val' or sp2 == '_trained' or sp2 == '':
            print('Model already trained. Please run testing instead of training.')
            return
        it = int(sp2)
        itmax = it if it > itmax else itmax
    if itmax: 
        sys.stdout.write ('Half-trained model found. Loading...\n')
        state = load_trainable_variables (model, config.modelfn+str(itmax), opt=optimizer)
    else:
        state = {}

    nmse_hist_val = state.get('nmse_hist_val', [])
    best_count = state.get('best_count', 0)
    i = state.get('i', 0)

    brk = False
    X_0=torch.zeros(1,1,config.D,config.M,config.N)
    U_0=torch.complex(torch.zeros(1,1,config.D,config.M,config.N),torch.zeros(1,1,config.D,config.M,config.N))
    Z_0=U_0
    Z_0=Z_0.cuda()
    X_0=X_0.cuda()
    U_0=U_0.cuda()
    while(i < config.maxit and not brk):
        y_, x_, y_val_, x_val_ = setup_input_sc (
            config.test, config.tbs, config.vbs, config.val_step, i,
            config.SNR, train_data, valid_data, config.mask, config.thetas
            )
        epoch = i // len(y_)
        vals = list(zip(y_val_, x_val_))
        for batch in zip(y_, x_):
            i += 1
            y, x = batch
            y, x = y.cuda(), x.cuda()
            optimizer.zero_grad()

            #if config.net == 'LISTA_gen' or 'LISTA_gen_align':
            #    label=x[:,:,0,:,:]
            #    loss=model(y,label)
            #if config.net == 'LISTA_swimTransformer':
            label=x
            loss = model(y, label)
            #else:
            #    loss = model(X_0,y,Z_0,U_0,x) #y,x-->(5,1,512,512) forward(self,X,Y,Z,U,targets):
            loss.backward()
            optimizer.step()
            nmse_tr = loss.detach() / x.mean()
            db_tr = 10. * torch.log10( nmse_tr )

            if i % config.val_step == 0:
                y_val, x_val = vals.pop()
                y_val, x_val = y_val.cuda(), x_val.cuda()

                if config.net=='LISTA_gen' or 'LISTA_gen_align':
                    x_val=x_val[:,:,0,:,:]
                    with torch.no_grad():
                        nmse_val = model( y_val, x_val) / x_val.mean()
                elif config.net=='LISTA_CNN3d' or 'WSA_VSI' or 'RATIR':
                    with torch.no_grad():
                        nmse_val = model(y_val, x_val)/ x_val.mean()
                else:
                    with torch.no_grad():
                        nmse_val = model(X_0, y_val, Z_0, U_0, x_val) / x_val.mean()

                if torch.isnan (nmse_val):
                    raise RuntimeError ('nmse is nan. exiting...')

                nmse_hist_val = np.append (nmse_hist_val, nmse_val.cpu())
                db_best_val = 10. * np.log10 (nmse_hist_val.min())
                db_val = 10. * torch.log10 (nmse_val)
                sys.stdout.write(
                        "\r| epoch={epoch:<7d} | i={i:<7d} | loss_tr={loss_tr:.6f} | "
                        "db_tr={db_tr:.6f}dB | loss_val={loss_val:.6f} | "
                        "db_val={db_val:.6f}dB | (best={db_best_val:.6f})"\
                            .format(epoch=epoch, i=i, loss_tr=nmse_tr,
                                    db_tr=db_tr, loss_val=nmse_val,
                                    db_val=db_val, db_best_val=db_best_val))
                sys.stdout.flush()
                if i % (10 * config.val_step) == 0:
                    print('')
                age_of_best = (len(nmse_hist_val) -
                               nmse_hist_val.argmin() - 1)
                # If nmse has not improved for a long time, stop training.
                if age_of_best * config.val_step > config.better_wait:
                    brk = True
                    break
                elif age_of_best == 0:
                    best_count += 1
                    if best_count >= 15:
                        best_count == 0
                        for fn in os.listdir(config.expbase):
                            if len(fn.split(config.model)) == 2:
                                os.remove(os.path.join(config.expbase, fn))
                    state['nmse_hist_val'] = nmse_hist_val
                    state['best_count'] = best_count
                    state['i'] = i
                    state['epoch'] = epoch
                    save_trainable_variables (model, config.modelfn + str(i), 
                                              opt=optimizer, **state)
                else:
                    best_count = 0

        del x_, y_
        if i % config.val_step == 0:
            del x_val_, y_val_



    print('')
    model.save_trainable_variables (config.modelfn + '_trained')
    model.load_trainable_variables (
        config.modelfn + str((nmse_hist_val.argmin() + 1) * config.val_step))
    model.save_trainable_variables (config.modelfn + '_best_val')


def save_trainable_variables (model, filename, **kwargs):
    """
    Save trainable variables in the model to npz file with current value of
    each variable in tf.trainable_variables().

    :sess: Tensorflow session.
    :filename: File name of saved file.
    :scope: Name of the variable scope that we want to save.
    :kwargs: Other arguments that we want to save.

    """
    save = dict ()
    save['state_dict'] = model.state_dict()

    if 'opt' in kwargs:
        opt = kwargs.pop('opt')
        save['opt_state_dict'] = opt.state_dict()

    # file name suffix check
    if not filename.endswith('.pt'):
        filename = filename + '.pt'

    save.update (kwargs)
    torch.save (save , filename)


def load_trainable_variables (model, filename, opt=None):
    """
    Load trainable variables from saved file.
    :sess: TODO
    :filename: TODO
    :returns: TODO

    """
    other = dict ()
    # file name suffix check
    if not filename.endswith('.pt'):
        filename = filename + '.pt'
    if not os.path.exists (filename):
        raise ValueError (filename + ' not exists')
    saved = torch.load (filename, 'cuda')
    assert not opt or 'opt_state_dict' in saved, \
        'Specified opt, but opt_state_dict not found'

    for k, d in saved.items ():
        if k == 'state_dict':
            print ('restoring state_dict of the model...')
            model.load_state_dict(d)
        elif k == 'opt_state_dict':
            if opt:
                print ('restoring state_dict of the optimizer...')
                opt.load_state_dict(d)
        else:
            other [k] = d

    return other

