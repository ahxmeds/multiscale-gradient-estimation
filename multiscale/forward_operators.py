import torch
import torch.nn as nn
import torch.nn.functional as F

class blurFFT(nn.Module):
    """Gaussian blur operator using FFT."""
    def __init__(self, dim=128, sigma=[3, 3], device='cuda'):
        super(blurFFT, self).__init__()
        self.dim = dim
        self.device = device
        self.sigma = sigma

        P, center = self.psfGauss(self.dim)
        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0, 1])).unsqueeze(0).unsqueeze(0)
        self.S = S.to(self.device)

    def forward(self, I):
        """Apply blur to image."""
        B = torch.real(torch.fft.ifft2(self.S * torch.fft.fft2(I)))
        return B

    def adjoint(self, Ic):
        """Apply adjoint blur operation."""
        I = self.forward(Ic)
        return I

    def psfGauss(self, dim):
        """Create Gaussian point spread function."""
        s = self.sigma
        m = dim
        n = dim

        x = torch.arange(-n//2+1, n//2+1)
        y = torch.arange(-n//2+1, n//2+1)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        PSF = torch.exp(-(X**2)/(2*s[0]**2) - (Y**2)/(2*s[1]**2))
        PSF = PSF / torch.sum(PSF)

        # Get center ready for output
        center = [1-m//2, 1-n//2]

        return PSF, center
