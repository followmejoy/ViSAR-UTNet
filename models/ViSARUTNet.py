from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

from utils.modules import Linear
from utils.modules import PosEncoding
from utils.layers import EncoderLayer, DecoderLayer, \
    WeightedEncoderLayer, WeightedDecoderLayer
from utils.complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexLinear
from utils.SARop import CSA_rangecompression, CSA_imag
from models.model_base_troch import model_base


def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i + 1].sum() - 1) / (i + 1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs - t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    # assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q, _ = seq_q.size()
    b_size, len_k, _ = seq_k.size()
    pad_attn_mask = seq_k.data.eq(seq_k)  # b_size x 1 x len_k
    return pad_attn_mask  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    # assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 dropout=0.01, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, return_attn=True):

        enc_outputs = self.dropout_emb(enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 dropout=0.01, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs, return_attn=True):
        dec_outputs = self.dropout_emb(dec_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.dropout, opt.weighted_model)
        self.decoder = Decoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.dropout, opt.weighted_model)
        self.tgt_proj = Linear(opt.d_model, opt.d_model, False)
        self.weighted_model = opt.weighted_model

        if opt.share_proj_weight:
            print('Sharing target embedding and projection..')
            self.tgt_proj.weight = self.decoder.tgt_emb.weight

        if opt.share_embs_weight:
            print('Sharing source and target embedding..')
            assert opt.src_vocab_size == opt.tgt_vocab_size, \
                'To share word embeddings, the vocabulary size of src/tgt should be the same'
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight

    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)

        return param_groups

    def encode(self, enc_inputs, return_attn=False):
        return self.encoder(enc_inputs, return_attn)

    def decode(self, dec_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, enc_outputs, return_attn)

    def forward(self, enc_inputs, dec_inputs, return_attn=False):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, return_attn)
        dec_outputs, dec_self_attns, dec_enc_attns = \
            self.decoder(dec_inputs, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
        # return dec_logits.view(-1, dec_logits.size(-1)), \
        #        enc_self_attns, dec_self_attns, dec_enc_attns

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass


class ViSARUTNet_Net(model_base):
    def __init__(self, opt, mask, thetas):
        super(ViSARUTNet_Net, self).__init__()
        self.transfor = Transformer(opt)
        self.loss = nn.MSELoss()

        self.cnn_encoder_complex = nn.Sequential(
            ComplexConv2d(1, 16, (7, 7), stride=1, padding=3),
            ComplexBatchNorm2d(16),
            # nn.MaxPool2d(2, 2),

            ComplexConv2d(16, 32, (5, 5), stride=1, padding=2),
            ComplexBatchNorm2d(32),
            # nn.MaxPool2d(2, 2),

            ComplexConv2d(32, 64, (3, 3), stride=1, padding=1),
            ComplexBatchNorm2d(64),
            # nn.MaxPool2d(2, 2),

            ComplexConv2d(64, 1, (1, 1), stride=1, padding=0),
            ComplexBatchNorm2d(1)
        )

        self.cnn_decoder_complex = nn.Sequential(
            ComplexConv2d(1, 64, (3, 3), stride=1, padding=1),
            ComplexBatchNorm2d(64),
            # nn.Upsample(scale_factor=2, mode='nearest'),

            ComplexConv2d(64, 32, (5, 5), stride=1, padding=2),
            ComplexBatchNorm2d(32),
            # nn.Upsample(scale_factor=2, mode='nearest'),

            ComplexConv2d(32, 16, (7, 7), stride=1, padding=3),
            ComplexBatchNorm2d(16),
            # nn.Upsample(scale_factor=2, mode='nearest'),

            ComplexConv2d(16, 1, (1, 1), stride=1, padding=0),
            ComplexBatchNorm2d(1)
        )

        if isinstance(mask, str):
            mask = loadmat(mask)  # 严谨起见可加文件是否存在的检测
            mask = torch.tensor(mask['mask'], dtype=torch.bool)
        elif isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool)
        elif not isinstance(mask, torch.Tensor):
            raise TypeError(f'Expected types of mask: {str}, {np.ndarray}, or {torch.Tensor}, got {type(mask)}')
        self.register_buffer('_mask', mask, False)  # 将一个张量注册为模型参数的方法,而这个参数不需要更新

        if isinstance(thetas, str):
            thetas = loadmat(thetas)
            Theta1 = torch.tensor(thetas['Theta1'], dtype=torch.complex64)
            Theta2 = torch.tensor(thetas['Theta2'], dtype=torch.complex64)
            Theta3 = torch.tensor(thetas['Theta3'], dtype=torch.complex64)
        elif isinstance(thetas, (tuple, list)):
            Theta1, Theta2, Theta3 = thetas
            Theta1 = torch.as_tensor(Theta1, dtype=torch.complex64)
            Theta2 = torch.as_tensor(Theta2, dtype=torch.complex64)
            Theta3 = torch.as_tensor(Theta3, dtype=torch.complex64)
        else:
            raise TypeError(f'Expected types of thetas: {str}, {tuple}, or {list}, got {type(thetas)}')
        self.register_buffer('_theta1', Theta1, False)
        self.register_buffer('_theta2', Theta2, False)
        self.register_buffer('_theta3', Theta3, False)
        self._thetas = [self._theta1, self._theta2, self._theta3]


    def inference(self, y):
        # echo = torch.view_as_complex(Y)
        echo = torch.mul(y, self._mask)
        RP = CSA_rangecompression(echo, self._thetas)
        dim = RP.shape
        image = CSA_imag(echo, self._thetas)
        RP_cat = RP.reshape(dim[0], dim[1], 1, dim[2]*dim[3], dim[4]) # cat data along frame dim
        RP2 = torch.cat((RP_cat.real, RP_cat.imag), dim=4)
        image_cat = image.reshape(dim[0], dim[1], 1, dim[2]*dim[3], dim[4]) # cat data along frame dim
        image2 = torch.cat((image_cat.real, image_cat.imag), dim=4)
        out_image, _, _, _ = self.transfor(RP2.squeeze(1).squeeze(1), image2.squeeze(1).squeeze(1))
        temp1 = out_image.unsqueeze(-1).reshape(dim[0], dim[2]*dim[3], dim[4], -1)
        temp2 = torch.abs(torch.view_as_complex(temp1))
        out_image2 = temp2.reshape(dim[0], dim[2], dim[3], dim[4])

        return out_image2.unsqueeze(1)

    def forward(self, y, label):
        x = self.inference(y)

        return self.loss(x, label)

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass



