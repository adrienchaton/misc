



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
from matplotlib import pyplot as plt



###############################################################################
###############################################################################
## DCT operators

# https://github.com/zh217/torch-dct
# https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


'''
###############################################################################
###############################################################################
## compute transient envelope

sample_rate = 16000
window_size = 160
dct_maxbin = window_size//4

file = '/Users/adrienbitton/Documents/acids/team/adrien/AE_DDSP/backup/dev_audio/drum_loop.wav'
x = librosa.core.load(file,sr=sample_rate,mono=True)[0]
n_wins = x.shape[0]//window_size
x = x[:n_wins*window_size]
x = torch.from_numpy(x).view(-1,window_size)

DCT_layer = LinearDCT(window_size,'dct')
iDCT_layer = LinearDCT(window_size,'idct')
X_dct = DCT_layer(x)
x_idct = iDCT_layer(X_dct)
# print(x.shape)
# print(X_dct.shape)
# print(torch.min(X_dct),torch.max(X_dct),torch.mean(X_dct))
# print(x_idct.shape)

X_fft = torch.rfft(x,1)
X_fft = torch.sqrt(X_fft[:,:,0]**2+X_fft[:,:,1]**2)
X_fft = torch.abs(X_fft)**2
X_fft = torch.log10(torch.clamp(X_fft,min=1e-7))*10.

plt.figure()
ax = plt.subplot(4,1,1)
ax.imshow(X_dct[:,:dct_maxbin].permute(1,0).numpy(),aspect='auto')
ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,2)
ax.plot(torch.sum(torch.abs(X_dct[:,:dct_maxbin]),1).numpy())
ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,3)
ax.imshow(X_fft.permute(1,0).numpy(),aspect='auto')
ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,4)
ax.plot(torch.sum(X_fft,1).numpy())
ax.set_xlim([0,n_wins-1])
plt.show()

X_dct_truncated = X_dct.clone()
X_dct_truncated[:,dct_maxbin:] = 0
x_idct_truncated = iDCT_layer(X_dct_truncated)
plt.figure()
plt.suptitle('output')
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_idct_truncated.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(x_idct_truncated.view(-1).numpy())
plt.show()
sf.write('./x_idct_truncated.wav',x_idct_truncated.view(-1).numpy(),sample_rate)

x_res = x.view(-1)-x_idct_truncated.view(-1)
plt.figure()
plt.suptitle('output')
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_res.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(x_res.view(-1).numpy())
plt.show()
sf.write('./x_res.wav',x_res.view(-1).numpy(),sample_rate)
'''


###############################################################################
###############################################################################
## regular HPSS operations (only harmo-percu)

# https://librosa.org/doc/latest/generated/librosa.decompose.hpss.html
# https://gist.github.com/keunwoochoi/dcbaf3eaa72ca22ea4866bd5e458e32c

def _enhance_either_hpss(x_padded, out, kernel_size, power, which, offset):
    """x_padded: one that median filtering can be directly applied
    kernel_size = int
    dim: either 2 (freq-axis) or 3 (time-axis)
    which: str, either "harm" or "perc"
    """
    if which == "harm":
        for t in range(out.shape[3]):
            out[:, :, :, t] = torch.median(x_padded[:, :, offset:-offset, t:t + kernel_size], dim=3)[0]

    elif which == "perc":
        for f in range(out.shape[2]):
            out[:, :, f, :] = torch.median(x_padded[:, :, f:f + kernel_size, offset:-offset], dim=2)[0]
    else:
        raise NotImplementedError("it should be either but you passed which={}".format(which))

    if power != 1.0:
        out.pow_(power)


def hpss(x, kernel_size=31, power=2.0, hard=False):
    """x: |STFT| (or any 2-d representation) in batch, (not in a decibel scale!)
    in a shape of (batch, ch, freq, time)
    power: to which the enhanced spectrograms are used in computing soft masks.
    kernel_size: odd-numbered {int or tuple of int}
        if tuple,
            1st: width of percussive-enhancing filter (one along freq axis)
            2nd: width of harmonic-enhancing filter (one along time axis)
        if int,
            it's applied for both perc/harm filters
    """
    eps = 1e-7
    if isinstance(kernel_size, tuple):
        pass
    else:
        # pad is int
        kernel_size = (kernel_size, kernel_size)

    pad = (kernel_size[0] // 2, kernel_size[0] // 2,
           kernel_size[1] // 2, kernel_size[1] // 2,)

    harm, perc, ret = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    x_padded = F.pad(x, pad=pad, mode='reflect')

    _enhance_either_hpss(x_padded, out=perc, kernel_size=kernel_size[0], power=power, which='perc', offset=kernel_size[1]//2)
    _enhance_either_hpss(x_padded, out=harm, kernel_size=kernel_size[1], power=power, which='harm', offset=kernel_size[0]//2)

    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)

    return x * mask_harm, x * mask_perc#, mask_harm, mask_perc



###############################################################################
###############################################################################
## extended HPSS operations (with residual)

def extended_hpss(x, kernel_size=21, beta=1.5, eps=1e-7):
    """
    receives x as magnitude spectrogram (batch,freq,time)
    or x as complex spectrogram (batch,freq,time,2)
    can be any representation, any amplitude scale
    the higher the beta, the stronger the contrast between harmo and percu components
    applies reflection padding of kernel_size//2 (kernel should be odd)
    can have different kernel_size for freq (e.g. 500Hz) and time frames (e.g. 200ms)
    kernel_size = int or [freq,time]
    """
    
    if type(kernel_size) is int:
        offset = kernel_size // 2
        pad = (offset, offset, offset, offset)
        
        if x.dim()==3:
            x_padded = F.pad(x.unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        if x.dim()==4:
            x_padded = F.pad(torch.sqrt(x[:,:,:,0]**2+x[:,:,:,1]**2).unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        
        harmo_median = torch.median(x_padded[:,offset:-offset,:].unfold(2, kernel_size, 1),-1)[0]
        percu_median = torch.median(x_padded[:,:,offset:-offset].unfold(1, kernel_size, 1),-1)[0]
    else:
        # harmo along time
        offset = kernel_size[1] // 2
        pad = (offset, offset, 0, 0) # left,right,top,bottom
        if x.dim()==3:
            x_padded = F.pad(x.unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        if x.dim()==4:
            x_padded = F.pad(torch.sqrt(x[:,:,:,0]**2+x[:,:,:,1]**2).unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        harmo_median = torch.median(x_padded.unfold(2, kernel_size[1], 1),-1)[0]
        # percu along freq
        offset = kernel_size[0] // 2
        pad = (0, 0, offset, offset)
        if x.dim()==3:
            x_padded = F.pad(x.unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        if x.dim()==4:
            x_padded = F.pad(torch.sqrt(x[:,:,:,0]**2+x[:,:,:,1]**2).unsqueeze(1), pad=pad, mode='reflect').squeeze(1)
        percu_median = torch.median(x_padded.unfold(1, kernel_size[0], 1),-1)[0]
    
    harmo_mask = torch.where(torch.gt(harmo_median/(percu_median+eps), beta),\
                             torch.ones_like(harmo_median), torch.zeros_like(harmo_median))
    percu_mask = torch.where(torch.gt(percu_median/(harmo_median+eps), beta),\
                             torch.ones_like(harmo_median), torch.zeros_like(harmo_median))
    
    residual_mask = torch.ones_like(harmo_median) - harmo_mask - percu_mask
    # assert (torch.any(residual_mask>=0)),"residual is not positive"
    
    if x.dim()==4:
        harmo_mask = harmo_mask.unsqueeze(-1).repeat(1,1,1,2)
        percu_mask = percu_mask.unsqueeze(-1).repeat(1,1,1,2)
        residual_mask = residual_mask.unsqueeze(-1).repeat(1,1,1,2)
    
    return x*harmo_mask, x*percu_mask, x*residual_mask


'''
###############################################################################
###############################################################################
## testing regular and extended HPSS

from nnAudio import Spectrogram
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

sample_rate = 16000
n_mels = 256
n_fft = 1024
hop_length = 256
kernel_size = 15
power = 2.0
hard = False

file = './backup/dev_audio/drum_loop.wav'
# file = './backup/dev_audio/drums.wav'
x = torch.from_numpy(librosa.core.load(file,sr=sample_rate,mono=True)[0]).unsqueeze(0).cuda()


mel_layer = Spectrogram.MelSpectrogram(sr=sample_rate,\
                            n_fft=n_fft,n_mels=n_mels,hop_length=hop_length,fmin=20.,power=1.0)
stft_layer = Spectrogram.STFT(sr=sample_rate,freq_scale='linear',\
                            n_fft=n_fft,hop_length=hop_length,fmin=20.)
log_stft_layer = Spectrogram.STFT(sr=sample_rate,freq_scale='log',\
                            n_fft=n_fft,hop_length=hop_length,fmin=20.)



## regular HPSS

mel = mel_layer(x).unsqueeze(1) # dummy channel for (batch, ch, freq, time)
stft = stft_layer(x).unsqueeze(1)
stft = torch.sqrt(stft[:,:,:,:,0]**2+stft[:,:,:,:,1]**2)
log_stft = log_stft_layer(x).unsqueeze(1)
log_stft = torch.sqrt(log_stft[:,:,:,:,0]**2+log_stft[:,:,:,:,1]**2)

mel_harm, mel_perc = hpss(mel, kernel_size=kernel_size, power=power, hard=hard)
stft_harm, stft_perc = hpss(stft, kernel_size=kernel_size, power=power, hard=hard)
log_stft_harm, log_stft_perc = hpss(log_stft, kernel_size=kernel_size, power=power, hard=hard)
# hpss output is power spectrogram already if power = 2.0

mel = torch.log10(torch.clamp(mel,min=1e-7))*10. # power to dB spectrogram
mel_harm = torch.log10(torch.clamp(mel_harm,min=1e-7))*10. # power to dB spectrogram
mel_perc = torch.log10(torch.clamp(mel_perc,min=1e-7))*10. # power to dB spectrogram

stft = torch.log10(torch.clamp(stft,min=1e-7))*10. # power to dB spectrogram
stft_harm = torch.log10(torch.clamp(stft_harm,min=1e-7))*10. # power to dB spectrogram
stft_perc = torch.log10(torch.clamp(stft_perc,min=1e-7))*10. # power to dB spectrogram

log_stft = torch.log10(torch.clamp(log_stft,min=1e-7))*10. # power to dB spectrogram
log_stft_harm = torch.log10(torch.clamp(log_stft_harm,min=1e-7))*10. # power to dB spectrogram
log_stft_perc = torch.log10(torch.clamp(log_stft_perc,min=1e-7))*10. # power to dB spectrogram


plt.figure(figsize=(18,12))
plt.suptitle('regular HPSS - ref. logspec - mel input - mel harm - mel perc')
plt.subplot(4,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(4,1,2)
ax.imshow(mel.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,3)
ax.imshow(mel_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,4)
ax.imshow(mel_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
plt.savefig('./mel_HPSS.png',format='png')
plt.close('all')

plt.figure(figsize=(18,12))
plt.suptitle('regular HPSS - ref. logspec - stft input - stft harm - stft perc')
plt.subplot(4,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(4,1,2)
ax.imshow(stft.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,3)
ax.imshow(stft_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,4)
ax.imshow(stft_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
plt.savefig('./stft_HPSS.png',format='png')
plt.close('all')

plt.figure(figsize=(18,12))
plt.suptitle('regular HPSS - ref. logspec - logstft input - logstft harm - logstft perc')
plt.subplot(4,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(4,1,2)
ax.imshow(log_stft.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,3)
ax.imshow(log_stft_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(4,1,4)
ax.imshow(log_stft_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
plt.savefig('./log_stft_HPSS.png',format='png')
plt.close('all')



## extended HPSS

# kernel_size = 15  # int or [freq,time]
kernel_size = [5,17]
beta = 1.5

mel = mel_layer(x)
stft = stft_layer(x)
log_stft = log_stft_layer(x)

stft = torch.sqrt(stft[:,:,:,0]**2+stft[:,:,:,1]**2)
log_stft = torch.sqrt(log_stft[:,:,:,0]**2+log_stft[:,:,:,1]**2)

mel_harm, mel_perc, mel_res = extended_hpss(mel, kernel_size=kernel_size, beta=beta)
stft_harm, stft_perc, stft_res = extended_hpss(stft, kernel_size=kernel_size, beta=beta)
log_stft_harm, log_stft_perc, log_stft_res = extended_hpss(log_stft, kernel_size=kernel_size, beta=beta)
# is not **pow

# mel = torch.log10(torch.clamp(mel**power,min=1e-7))*10.
mel_harm = torch.log10(torch.clamp(mel_harm**power,min=1e-7))*10.
mel_perc = torch.log10(torch.clamp(mel_perc**power,min=1e-7))*10.
mel_res = torch.log10(torch.clamp(mel_res**power,min=1e-7))*10.

# stft = torch.log10(torch.clamp(stft**power,min=1e-7))*10.
stft_harm = torch.log10(torch.clamp(stft_harm**power,min=1e-7))*10.
stft_perc = torch.log10(torch.clamp(stft_perc**power,min=1e-7))*10.
stft_res = torch.log10(torch.clamp(stft_res**power,min=1e-7))*10.

# log_stft = torch.log10(torch.clamp(log_stft**power,min=1e-7))*10.
log_stft_harm = torch.log10(torch.clamp(log_stft_harm**power,min=1e-7))*10.
log_stft_perc = torch.log10(torch.clamp(log_stft_perc**power,min=1e-7))*10.
log_stft_res = torch.log10(torch.clamp(log_stft_res**power,min=1e-7))*10.

mel = mel_harm+mel_perc+mel_res
stft = stft_harm+stft_perc+stft_res
log_stft = log_stft_harm+log_stft_perc+log_stft_res


plt.figure(figsize=(18,12))
plt.suptitle('ext. HPSS - ref. logspec - mel sum - mel harm - mel perc - mel res')
plt.subplot(5,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(5,1,2)
ax.imshow(mel.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,3)
ax.imshow(mel_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,4)
ax.imshow(mel_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,5)
ax.imshow(mel_res.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.savefig('./mel_HPSSext.png',format='png')
plt.close('all')

plt.figure(figsize=(18,12))
plt.suptitle('ext. HPSS - ref. logspec - stft sum - stft harm - stft perc - stft res')
plt.subplot(5,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(5,1,2)
ax.imshow(stft.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,3)
ax.imshow(stft_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,4)
ax.imshow(stft_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,5)
ax.imshow(stft_res.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.savefig('./stft_HPSSext.png',format='png')
plt.close('all')

plt.figure(figsize=(18,12))
plt.suptitle('ext. HPSS - ref. logspec - log_stft sum - log_stft harm - log_stft perc - log_stft res')
plt.subplot(5,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
ax = plt.subplot(5,1,2)
ax.imshow(log_stft.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,3)
ax.imshow(log_stft_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,4)
ax.imshow(log_stft_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
# plt.colorbar()
# ax.set_xlim([0,n_wins-1])
ax = plt.subplot(5,1,5)
ax.imshow(log_stft_res.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.savefig('./log_stft_HPSSext.png',format='png')
plt.close('all')



## extended HPSS on complex spectrogram for resynthesis
# nnAudio iSTFT is not inverting properly
# can do with torch stft and istft

import soundfile as sf

stft = torch.stft(x, n_fft, hop_length=hop_length, win_length=n_fft,\
    window=torch.hann_window(n_fft).cuda(), center=True, pad_mode='reflect', normalized=False, onesided=True)

stft_harm, stft_perc, stft_res = extended_hpss(stft, kernel_size=kernel_size, beta=beta)

x_harm = torch.istft(stft_harm, n_fft, hop_length=hop_length, win_length=n_fft,\
    window=torch.hann_window(n_fft).cuda(), center=True, normalized=False, onesided=True, length=None)
x_perc = torch.istft(stft_perc, n_fft, hop_length=hop_length, win_length=n_fft,\
    window=torch.hann_window(n_fft).cuda(), center=True, normalized=False, onesided=True, length=None)
x_res = torch.istft(stft_res, n_fft, hop_length=hop_length, win_length=n_fft,\
    window=torch.hann_window(n_fft).cuda(), center=True, normalized=False, onesided=True, length=None)

x_sum = x_harm+x_perc+x_res
stft_sum = stft_harm+stft_perc+stft_res

stft_harm = torch.sqrt(stft_harm[:,:,:,0]**2+stft_harm[:,:,:,1]**2)
stft_perc = torch.sqrt(stft_perc[:,:,:,0]**2+stft_perc[:,:,:,1]**2)
stft_res = torch.sqrt(stft_res[:,:,:,0]**2+stft_res[:,:,:,1]**2)
stft_sum = torch.sqrt(stft_sum[:,:,:,0]**2+stft_sum[:,:,:,1]**2)

stft_harm = torch.log10(torch.clamp(stft_harm**power,min=1e-7))*10.
stft_perc = torch.log10(torch.clamp(stft_perc**power,min=1e-7))*10.
stft_res = torch.log10(torch.clamp(stft_res**power,min=1e-7))*10.
stft_sum = torch.log10(torch.clamp(stft_sum**power,min=1e-7))*10.

plt.figure(figsize=(18,12))
plt.suptitle('comp. HPSS - stft sum - stft harm - stft perc - stft res')
plt.subplot(4,1,1)
plt.imshow(stft_sum.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.subplot(4,1,2)
plt.imshow(stft_harm.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.subplot(4,1,3)
plt.imshow(stft_perc.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.subplot(4,1,4)
plt.imshow(stft_res.squeeze().flip(0).cpu().numpy(),aspect='auto')
plt.savefig('./stft_HPSSext_complex.png',format='png')
plt.close('all')

plt.figure(figsize=(18,12))
plt.suptitle('inverted comp. HPSS - x sum - x harm - x perc - x res')
plt.subplot(4,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_sum.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
plt.subplot(4,1,2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_harm.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
plt.subplot(4,1,3)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_perc.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
plt.subplot(4,1,4)
D = librosa.amplitude_to_db(np.abs(librosa.stft(x_res.view(-1).cpu().numpy(),n_fft=n_fft,hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='log',sr=sample_rate,hop_length=hop_length)
plt.savefig('./stft_HPSSext_complex_inverted.png',format='png')
plt.close('all')

sf.write('./stft_HPSSext_harm.wav',x_harm.view(-1).cpu().numpy(),sample_rate)
sf.write('./stft_HPSSext_perc.wav',x_perc.view(-1).cpu().numpy(),sample_rate)
sf.write('./stft_HPSSext_res.wav',x_res.view(-1).cpu().numpy(),sample_rate)
sf.write('./stft_HPSSext_sum.wav',x_sum.view(-1).cpu().numpy(),sample_rate)
'''

