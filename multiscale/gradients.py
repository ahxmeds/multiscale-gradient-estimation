import torch
import torch.nn as nn
import torch.nn.functional as F

# Multiscale decomposition utility
def ms_decompose(x, scales=5):
    xS = [x]
    for _ in range(scales):
        xSi = F.interpolate(xS[0], scale_factor=0.5, mode='bilinear', align_corners=True)
        xS.insert(0, xSi)
    return xS

# Example multiscale convolution block
class MSconv(nn.Module):
    def __init__(self, in_c=3, out_c=64):
        super().__init__()
        scale = out_c * in_c
        coef = torch.randn(out_c, in_c, 3, 3)
        coef = coef - coef.mean(dim=[2, 3], keepdim=True)
        self.coef = nn.Parameter(1/scale * coef)
        # Dummy A matrices for demonstration
        self.ThH = torch.eye(9)

    def get_conv_kernels(self, coef, levels):
        W = coef.reshape(coef.shape[0], coef.shape[1], 9)
        for _ in range(levels):
            W = W @ self.ThH
        W = W.reshape(coef.shape[0], coef.shape[1], 3, 3)
        return W

    def forward(self, u, levels=0):
        W = self.get_conv_kernels(self.coef, levels)
        out = F.conv2d(u, W, padding=1)
        return out

# Example spectral convolution block
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, out_ft.shape[-2]//2)
        m2 = min(self.modes2, out_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, -m1:, :m2])
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# Example NFO layer using spectral convs
class NFO_layer(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, modes1=8, modes2=8):
        super().__init__()
        self.Conv1 = SpectralConv2d(in_channels, hid_channels, modes1, modes2)
        self.Conv2 = SpectralConv2d(hid_channels, out_channels, modes1, modes2)
    def forward(self, x):
        x = self.Conv1(x)
        x = F.layer_norm(x, (x.shape[1], x.shape[2], x.shape[3]))
        x = F.tanh(x)
        x = self.Conv2(x)
        return x
