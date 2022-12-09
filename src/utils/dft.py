import essentia.standard as es
import numpy as np



def compute_spectrogram(x, fft_size=1024, hop_size=1024, window_type='square'):
    """ Gets the spectrogram of waveform x
    
    Args:
        x (numpy.array): Array of samples
        fft_size (int): FFT size
        hop_size (int): Hop size between windows
        window_type (str): Window type (e.g. 'square' or 'hann')

    Returns:
        (np.array): Spectrogram with shape (num_frames, fft_size / 2 + 1) 
    """
    w = es.Windowing(type=window_type)
    fft = es.FFT()
    spectrogram = []
    for frame in es.FrameGenerator(x,
                                   frameSize=fft_size,
                                   hopSize=hop_size,
                                   startFromZero=True):
        windowed_frame = w(frame)
        fft_windowed_frame = fft(windowed_frame)
        spectrogram.append(fft_windowed_frame)
    spectrogram = np.array(spectrogram)

    # We do this to use the full dynamic range before quantization:
    spectrogram /= np.max(np.abs(spectrogram))

    return spectrogram

def compute_inverse_spectrogram(X, fft_size=1024, hop_size=1024):
    """ Gets the waveform from a spectrogram X
    
    Args:
        x (numpy.array): Array of samples
        fft_size (int): FFT size
        hop_size (int): Hop size between windows
        window_type (str): Window type (e.g. 'square' or 'hann')

    Returns:
        (np.array): Spectrogram with shape (num_frames, fft_size / 2 + 1) 
    """
    overlap_add = es.OverlapAdd(frameSize=fft_size, hopSize=hop_size)
    ifft = es.IFFT()
    y = np.array([], dtype=np.float32)
    for fft_windowed_frame in X:
        windowed_frame = ifft(fft_windowed_frame)
        frame_overlapped = overlap_add(windowed_frame)
        y = np.append(y, frame_overlapped)
    return y