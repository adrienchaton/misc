import numpy as np
import torch
import scipy
import librosa
import librosa.display
from matplotlib import pyplot as plt
import torchaudio

def remove_above_nyquist(frequency_envelopes,amplitude_envelopes,sample_rate):
    """
    frequency_envelopes: [batch_size, n_samples, n_sinusoids] (>=0)
    amplitude_envelopes: [batch_size, n_samples, n_sinusoids] (>=0)
    """
    amplitude_envelopes = torch.where(
    torch.gt(frequency_envelopes, sample_rate / 2.0),
            torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
    # note: should be greater or equal
    return amplitude_envelopes

def customwave_synth(f0_envelope,amplitude_envelope,sample_rate,overtones,mode,duty=0.2):
    """
    f0_envelope: [batch_size, n_samples, 1] (>=0)
    amplitude_envelope: [batch_size, n_samples, 1] (>=0)
    sawtooth = all overtones up to Nyquist with linear decay amplitude
    square = all odd overtones up to Nyquist with linear decay amplitude
    pulse = fourier expansion with duty cycle https://en.wikipedia.org/wiki/Pulse_wave
    e.g. f0=20Hz *400 (overtones) = 8000Hz (Nyquist)
    f0 should be pre-scaled in range [20,1200]
    """
    bs = f0_envelope.shape[0]
    n_overtones = overtones.shape[-1]
    frequency_envelopes = f0_envelope.expand(bs,-1,n_overtones)*overtones
    if mode=='sawtooth' or mode=='square' or mode=='pulse':
        amplitude_envelope = amplitude_envelope.expand(bs,-1,n_overtones)/overtones
    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(frequency_envelopes,amplitude_envelope,sample_rate)
    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample
    # Accumulate phase and synthesize.
    phases = torch.cumsum(omegas, axis=1)
    if mode=='sawtooth' or mode=='square':
        wavs = torch.sin(phases)
        audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
        audio = torch.sum(audio, axis=-1)  # [mb, n_samples]
        if mode=='sawtooth':
            audio = audio/2 # empirically it seems to give a good amplitude in [-0.9,0.9] in f0 range [40,400]
        if mode=='square':
            audio = audio
    if mode=='pulse':
        wavs = torch.cos(phases)
        audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
        audio = audio * 2.0 / np.pi
        audio = audio * torch.sin(duty*np.pi*overtones)
        audio = torch.sum(audio, axis=-1)+duty
    return audio


sample_rate = 16000
bs = 100
window_size = 640
length = bs*window_size
n_wins = length//window_size
fc = 2000.
Q = 1.


f0_min = 1000.
f0_max = 5000.
n_overtones = 400
f0 = np.linspace(f0_min,f0_max,num=length)
f0_envelope = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1).float()
amplitude_envelope = torch.ones(1,length,1)

overtones_saw = torch.arange(1,n_overtones+1).reshape(1,1,-1).float()
audio = customwave_synth(f0_envelope,amplitude_envelope,sample_rate,overtones_saw,'sawtooth')

## source audio
plt.figure()
plt.suptitle('source audio')
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(audio.view(-1).numpy())
plt.show()


## filtering audio in full-length
filtered_audio_1 = torchaudio.functional.lowpass_biquad(audio,sample_rate, fc, Q)
filtered_audio_2 = torchaudio.functional.lowpass_biquad(filtered_audio_1,sample_rate, fc, Q)
filtered_audio_3 = torchaudio.functional.lowpass_biquad(filtered_audio_2,sample_rate, fc, Q)

plt.figure()
plt.suptitle('BIQUAD filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_1.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_1.view(-1).numpy())
plt.show()

plt.figure()
plt.suptitle('BIQUADx2 filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_2.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_2.view(-1).numpy())
plt.show()

plt.figure()
plt.suptitle('BIQUADx3 filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_3.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_3.view(-1).numpy())
plt.show()


## filtering audio chunks of window_size
audio = audio.view(bs,window_size)

filtered_audio_1 = torchaudio.functional.lowpass_biquad(audio,sample_rate, fc, Q)
filtered_audio_2 = torchaudio.functional.lowpass_biquad(filtered_audio_1,sample_rate, fc, Q)
filtered_audio_3 = torchaudio.functional.lowpass_biquad(filtered_audio_2,sample_rate, fc, Q)

plt.figure()
plt.suptitle('BIQUAD filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_1.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_1.view(-1).numpy())
plt.show()

plt.figure()
plt.suptitle('BIQUADx2 filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_2.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_2.view(-1).numpy())
plt.show()

plt.figure()
plt.suptitle('BIQUADx3 filtering LPF at '+str(fc))
plt.subplot(2,1,1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio_3.view(-1).numpy(),n_fft=1024)),ref=np.max)
librosa.display.specshow(D, sr=sample_rate, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2,1,2)
plt.plot(filtered_audio_3.view(-1).numpy())
plt.show()