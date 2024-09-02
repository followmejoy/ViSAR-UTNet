'''
@Description: The implimention of Video SAR Imaging based on Unfolded Transformer
@Author: LiMin
@Date: 2024-05-10
'''

from scipy.io import loadmat
import numpy as np
import torch
from torch import nn
from utils.SARop import CSA_imag, CSA_echo
from models.model_base_troch import model_base
from models.Swimtransformer import PatchEmbed,BasicLayer
from timm.models.layers import  to_2tuple


class Weighted_SA(nn.Module):
    """
    input:C*D*H*W
    C:channel number
    D:depth, (the CT slices number)
    H*W: height and width of one image
    """

    def __init__(self, in_ch, out_ch, D, W, H):
        super().__init__()
        self.C = in_ch
        self.D = D
        self.H = H
        self.W = W
        self.gama = nn.Parameter(torch.tensor([0.0]))

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(self.in_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.in_ch),
            nn.ReLU(inplace=True),
        )
        self.WSoftmax = WeightedSoftmax()

    def Cal_Patt(self, k_x, q_x, v_x, C, D, H, W):
        """
        input:C*D*H*W
        """
        k_x_flatten = k_x.reshape((C, D, -1, H * W))
        q_x_flatten = q_x.reshape((C, D, -1, H * W))
        v_x_flatten = v_x.reshape((C, D, -1, H * W))
        sigma_x = torch.mul(q_x_flatten, k_x_flatten)
        r_x = self.WSoftmax(sigma_x, 3)  # dim=4
        # r_x = F.softmax(sigma_x, dim=4)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Patt = torch.mul(v_x_flatten, r_x).reshape(-1, C, D, H, W)
        return Patt

    def Cal_Datt(self, k_x, q_x, v_x):
        """
        input:C*D*H*W
        """
        # k_x_transpose = k_x.permute(0, 1, 3, 4, 2)
        # q_x_transpose = q_x.permute(0, 1, 3, 4, 2)
        # v_x_transpose = v_x.permute(0, 1, 3, 4, 2)
        k_x_flatten = k_x.permute(0, 1, 3, 4, 2)
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2)
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2)
        sigma_x = torch.mul(q_x_flatten, k_x_flatten)
        r_x = self.WSoftmax(sigma_x, 4)  # dim=5
        # r_x = F.softmax(sigma_x, dim=5)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Datt = torch.mul(v_x_flatten, r_x)
        return Datt.permute(0, 1, 4, 2, 3)

    def forward(self, x):
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)

        Patt = self.Cal_Patt(k_x, q_x, v_x, self.C, self.D, self.H, self.W)
        Datt = self.Cal_Datt(k_x, q_x, v_x)

        Y = self.gama * (Patt + Datt) + x
        return Y


class WeightedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, d):
        # Calculate the L2 norm of each frame
        norms = torch.norm(x, p=2, dim=[-2, -1],
                           keepdim=True)  # the last two dimensions indicate the image of each frame

        # Calculate the weights as the reciprocal of the norms
        weights = torch.exp(-1 / 2 * norms)

        # Apply element-wise multiplication of input and weights
        weighted_input = x * weights

        # Apply softmax function to the weighted input
        softmax_output = torch.softmax(weighted_input, dim=d)

        return softmax_output


class TSA_Layer(nn.Module):
    def __init__(self, heads, N_frames, W, H):
        super(TSA_Layer, self).__init__()
        self.in_ch = heads
        self.out_ch = 1
        self.N = N_frames
        self.D = N_frames
        self.W = W
        self.H = H
        self.conv3d_in = nn.Sequential(
            # Conv3d input:1*D*H*W
            # Conv3d output:C*D*H*W
            nn.Conv3d(in_channels=1, out_channels=self.in_ch, kernel_size=(1, 1, 1), padding=0),
            nn.BatchNorm3d(self.in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3d_out = nn.Sequential(
            # Conv3d input:1*D*H*W
            # Conv3d output:C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=1, kernel_size=(1, 1, 1), padding=0),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.WSA = Weighted_SA(self.in_ch, self.out_ch, self.D, self.W, self.H)

    def forward(self, x):
        # x = x.squeeze(0)
        x = self.conv3d_in(x)
        x = self.WSA(x)
        x = self.conv3d_out(x)
        # x = x.unsqueeze(0)
        return x


class NonLinear (nn.Module):
    def __init__(self, ps, qs):
        super().__init__()
        self.qs = nn.Parameter(qs)
        self.register_buffer('ps', ps, False)

    @staticmethod
    def _shrink_piecewise(r_, ps_, qs_):
        assert r_.dim() == 4, "Input must be a 4-dim tensor(NCHW)."
        channel, H, W = r_.shape[1:]
        assert channel == qs_.shape[-1], \
            f"Expected channels = {qs_.shape[-1]}, got {channel}."

        def expand2dchw(tensor):
            return tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        #增维 torch.unsqueeze(input, dim, out=None)增加大小为1的维度，也就是返回一个新的张量，对输入的指定位置插入维度 1且必须指明维度
        #expand （）函数expand函数的功能就是用来扩展张量中某维数据的尺寸，它返回输入张量在某维扩展为更大尺寸后的张量
        with torch.no_grad():#包装器“ with torch.no_grad（）”将所有require_grad标志临时设置为false。；不希望PyTorch计算新定义的变量param的梯度（减少计算量），因为他只想更新它们的值。
            k_ = torch.div(r_ - ps_[0], ps_[1] - ps_[0]).floor().cpu()
        idxmin_ = (k_ < 0.0).float()
        idxmax_ = (k_ >= ps_.shape[0] - 1.).float()
        idxmid_ = (1.0 - idxmin_ - idxmax_).cuda()
        k_ = (idxmid_ * k_.cuda()).long()
        out_ = idxmin_.cuda() * (r_ - ps_[0] + expand2dchw(qs_[0].unsqueeze(0)))
        out_ += idxmax_.cuda() * (r_ - ps_[-1] + expand2dchw(qs_[-1].unsqueeze(0)))
        qk_ = idxmid_ * torch.gather(expand2dchw(qs_), 0, k_)
        qkp1_ = idxmid_ * torch.gather(expand2dchw(qs_), 0, k_ + 1)
        pk_ = idxmid_ * torch.gather(expand2dchw(ps_.unsqueeze(-1).expand(-1, channel)), 0, k_)
        out_ += (qkp1_ - qk_) / (ps_[1] - ps_[0]) * (torch.multiply (idxmid_, r_) - pk_) + qk_
        return out_

    def forward(self, input):
        return checkpoint(NonLinear._shrink_piecewise, input, self.ps, self.qs)


class ISTA_Layer(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.sigma = nn.Parameter(torch.zeros(L))
        self.miu = nn.Parameter(torch.ones(L))
        self.L = L
        self.prox = nn.LeakyReLU()

    def shirnk_func(self, x, k):
        m = nn.LeakyReLU()
        return torch.sgn(x) * m(x.abs() - k)

    def forward(self, y, mask, thetas, z):
        # for i in range (self.L):
        #     temp1 = self.miu[i] * mask * CSA_imag(y, thetas)
        #     temp2 = self.miu[i] * CSA_imag(mask * CSA_echo(z, thetas), thetas)
        #     x = self.shirnk_func(z + temp1.abs() - temp2.abs(), self.sigma[i])
        #     z = self.proxfun(x)
        for i in range(self.L):
            # R_prime = y - CSA_echo(z, thetas) * mask
            R_prime = y - CSA_echo(z, thetas)
            delta_X = CSA_imag(R_prime, thetas)
            temp1 = torch.max(z.abs(), dim=-1)
            temp2 = torch.max(temp1.values, dim=-1)
            temp3 = temp2.values
            x = self.shirnk_func(z + self.miu[i] * delta_X.abs(), self.sigma[i] * (temp3.unsqueeze(-1).unsqueeze(-1)))
            z = self.prox(x)

        return x


class PatchUnembed(nn.Module):
    '''解卷积操作，用于保持数据维度一致'''
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.unproj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(-1, self.embed_dim, *self.patches_resolution)  # B Ph*Pw C
        x = self.unproj(x)
        return x


class SwimTransformerbolck(nn.Module):
    '''Swim Transformer中的一个一个完整的Block，在结尾添加解卷积操作保持数据维度一致'''
    def __init__(self, img_size=512, patch_size=4, in_chans=4, embed_dim=96, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.patch_norm= patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        #num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.SwimTransformer = BasicLayer(dim=embed_dim, input_resolution=(self.patches_resolution[0],self.patches_resolution[1]),
                                          depth=2, num_heads=4, window_size=8,
                                          mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                          drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                                          fused_window_process=False)
        self.patch_unembed=PatchUnembed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=None)

    def forward(self, x):
        x=self.patch_embed(x)
        x=self.SwimTransformer(x)
        x=self.patch_unembed(x)
        return x


class WSA_VSI(model_base):
    def __init__(self, iters, N_frames, W, H, mask, thetas):
        super(WSA_VSI, self).__init__()

        # Define the layers and parameters of the model
        self.L = iters
        self.ISTA = ISTA_Layer(1)
        self.N_heads = 8
        self.TSA = TSA_Layer(self.N_heads, N_frames, W, H)
        # self.TransformBlock = SwimTransformerbolck()
        self.loss = nn.MSELoss()
        self.D = N_frames
        self.W = W
        self.H = H

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
        # Define the forward pass of the model

        x = CSA_imag(y, self._thetas)
        x = x.abs()
        for i in range(self.L):
            z = self.TSA(x)
            # z = self.TransformBlock(x.squeeze(2))
            x = self.ISTA(y, self._mask, self._thetas, z.unsqueeze(2))
        # x = nn.functional.normalize(x.reshape(-1, self.D * self.W * self.H),
        #                             float('inf')).reshape(-1, 1, self.D, self.W, self.H)
        x = nn.functional.normalize(x.reshape(-1, self.W * self.H), p=2, dim=-1).reshape(-1, 1, self.D, self.W, self.H)

        return x

    def forward(self, y, label):
        x = self.inference(y)

        return self.loss(x, label)
