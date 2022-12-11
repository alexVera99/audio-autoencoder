from matplotlib import pyplot as plt
import torch
import numpy as np
from IPython.display import Audio
from IPython.display import display


from src.utils.dft import compute_inverse_spectrogram


def polar_2_rectangular(magnitude, phase):
    """Convert from a complex number from polar form \
        to rectangular form (i.e., a + b*j)-

    Args:
        magnitude (float): magnitude of the complex number
        phase (float): angle in radians of the complex number

    Returns:
        _type_: _description_
    """
    return magnitude * np.exp(1j*phase)

def batch_dft_to_audio(batch: torch.Tensor,
                       fft_size=1024, 
                       hop_size=1024,
                       listen=False,
                       listen_max = 5,
                       sr = 44100,
                       show_spectrogram=False) -> list:
    """Convert a batch of DFT in polar forma to audio signal

    Args:
        batch (torch.Tensor): batch of dfts in the form [B, C, FB, FR] \
            where B is the batch size; C is the number of channels \
            (in this case, the first channel is for the magnitude, \
            and the second channel, for the phase); FB is the number \
            of frequency bins and; FR is the number of frames.
        fft_size (int, optional): Size of the DFT. Defaults to 1024.
        hop_size (int, optional): Hop size of the DFT. Defaults to 1024.
        listen (boolean, optional): It displays a player to reproduce the \
            listen_max audios in the batch.
        listen_max (int): Maximum number of audios to listen. It will list the \
            first listen_max audios.
        sr (int, optional): Sampling rate to reproduce the audio when \
            listen==True. 
        show_spectrogram (boolean, optional): It plots the spectrogram of \
            listen_max audios in the batch.
    Return:
        A list of all the audios.
    """
    batch_np = batch.permute(0,2,3,1).detach().numpy()
    
    audios = list()
    
    for i, _b in enumerate(batch_np):
        magnitude = _b[:, :, 0].T
        phase = _b[:, :, 1].T
        
        dft = polar_2_rectangular(magnitude,
                                  phase)
        
        audio = compute_inverse_spectrogram(dft, 
                                            fft_size=fft_size,
                                            hop_size=hop_size)
        
        audios.append(audio)
        
        if i < listen_max:
            if listen:
                display(Audio(audio, rate=sr))
            if show_spectrogram:
                plot_spectrogram(dft)
    
    return audios


def plot_loss_function(losses_list, title = "Loss function"):
    plt.title(title)
    plt.plot(range(1,len(losses_list)+1), losses_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_spectrogram(X):
    plt.figure()
    plt.imshow(10*np.log10(np.abs(X) + 1e-6).T, origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.title('10 x log10(abs(X))')
    plt.show()