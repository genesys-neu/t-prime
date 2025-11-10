import torch
from torch import nn, Tensor
from preprocessing.RMS_power_norm import rms
import numpy as np

class RMSNorm(nn.Module):

    def __init__(self, model='Transformer'):
        super(RMSNorm, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert((model == 'Transformer') or (model == 'CNN'))
        self.model = model

    def forward(self, x: Tensor):
        if self.model == 'Transformer':
            x_shape = x.shape
            # here we have the input data in the form IQIQIQ...
            x_real = x[:, :, ::2]
            x_imag = x[:, :, 1::2]
            x_cpx = x_real + 1j * x_imag
            # 10 * np.log10(rms(x_cpx.cpu().numpy()) ** 2) # power before RMS norm
            rms_val = torch.sqrt(torch.mean(torch.mul(x_cpx.real, x_cpx.real) + torch.mul(x_cpx.imag, x_cpx.imag)))
            x_RMSnorm = x_cpx / rms_val
            # 10 * np.log10( rms(x_RMSnorm.cpu().numpy())** 2 ) # power after RMS norm
            x_out = torch.zeros(x_shape, dtype=x.dtype).to(self.device)
            x_out[:, :, ::2] = x_RMSnorm.real
            x_out[:, :, 1::2] = x_RMSnorm.imag

        else:
            x_out = x
            pass    # TODO cnn case

        return x_out

