'''
@Description: The implimention of SACNN, https://github.com/guoyii/SACNN/tree/master
@Author: LiMin
@Date: 2024-05-7
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.SARop import CSA_imag, CSA_echo


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


class Conv_3d(nn.Module):
    """
    input:N*C*D*H*W
    """

    def __init__(self, in_ch, out_ch, use_bn="use_bn"):
        super().__init__()
        if use_bn == "use_bn":
            self.conv3d = nn.Sequential(
                # Conv3d input:N*C*D*H*W
                # Conv3d output:N*C*D*H*W
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv3d(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


## Self-Attention Block
##***********************************************************************************************************

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
            # Conv3d input:C*D*H*W
            # Conv3d output:C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3d_1 = nn.Sequential(
            # Conv3d input:C*D*H*W
            # Conv3d output:C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )
        self.WSoftmax = WeightedSoftmax()

    def Cal_Patt(self, k_x, q_x, v_x, C, D, H, W):
        """
        input:C*D*H*W
        """
        k_x_flatten = k_x.reshape((C, D, 1, H * W))
        q_x_flatten = q_x.reshape((C, D, 1, H * W))
        v_x_flatten = v_x.reshape((C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 3, 2), k_x_flatten)
        r_x = self.WSoftmax(sigma_x, 3)  # dim=4
        # r_x = F.softmax(sigma_x, dim=4)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(C, D, H, W)
        return Patt

    def Cal_Datt(self, k_x, q_x, v_x, C, D, H, W):
        """
        input:C*D*H*W
        """
        # k_x_transpose = k_x.permute(0, 1, 3, 4, 2)
        # q_x_transpose = q_x.permute(0, 1, 3, 4, 2)
        # v_x_transpose = v_x.permute(0, 1, 3, 4, 2)
        k_x_flatten = k_x.permute(0, 2, 3, 1).reshape((C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 2, 3, 1).reshape((C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 2, 3, 1).reshape((C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 4, 3), k_x_flatten)
        r_x = self.WSoftmax(sigma_x, 4)  #dim=5
        # r_x = F.softmax(sigma_x, dim=5)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(C, H, W, D)
        return Datt.permute(0, 3, 1, 2)

    def forward(self, x):
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)

        Patt = self.Cal_Patt(k_x, q_x, v_x, self.C, self.D, self.H, self.W)
        Datt = self.Cal_Datt(k_x, q_x, v_x, self.C, self.D, self.H, self.W)

        Y = self.gama * (Patt + Datt) + x
        return Y


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
            nn.BatchNorm3d(heads),
            nn.ReLU(inplace=True),
        )
        self.WSA = Weighted_SA(self.in_ch, self.out_ch, self.D, self.W, self.H)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv3d_in(x)
        x = self.WSA(x)
        x = x.squeeze(0)
        return x


def shirnk_func(x, k):
    m = nn.LeakyReLU()
    return torch.sgn(x) * m(x.abs() - k)


class ISTA_Layer(nn.Module):
    def __int__(self):
        super().__init__()
        # self.b = nn.Parameter(0.1*torch.ones_like(mask))
        self.sigma = nn.Parameter(torch.tensor([0.0]))
        self.miu = nn.Parameter(torch.tensor([0.0]))

    def forward(self, y, mask, thetas, z=None):
        if z is None:
            x = CSA_imag(y, thetas)
        else:
            temp1 = self.miu * mask * CSA_imag(y, thetas)
            temp2 = self.miu * CSA_imag(mask * CSA_echo(z, thetas), thetas)
            x = shirnk_func(z + temp1.abs() - temp2.abs(), self.sigma)

        return x.abs()
