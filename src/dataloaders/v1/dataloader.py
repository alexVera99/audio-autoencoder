#Making native class loader
import numpy as np
import pandas as pd
import torch
import essentia.standard as es
from src.utils.dft import compute_spectrogram


class AudioDB(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self, data_filename, 
                 data_path,
                 transform = None,
                 fs = 44100,
                 fft_size=1024,
                 hop_size=1024, 
                 window_type='square'):
        self.audio_df = pd.read_csv(data_filename, sep=";")
        self.audio_df = self.audio_df.sample(frac=1, 
                                             ignore_index=True)

        self.data_path = data_path
        self.transform = transform
        self.fs = fs
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_type = window_type

    # What to do to load a single item in the dataset ( read image and label)    
    def __getitem__(self, index):
        sample_data = self.audio_df.iloc[index]

        f = self.data_path / sample_data["name"]
        start_time = int(sample_data["start_time"])
        end_time = int(sample_data["end_time"])
        
        audio_data = es.EasyLoader(filename=str(f),
                                   sampleRate=self.fs,
                                   startTime=start_time, 
                                   endTime=end_time)()
        
        dft = compute_spectrogram(audio_data,
                                  self.fft_size,
                                  self.hop_size,
                                  self.window_type)

        magnitud = np.abs(dft).T
        phase = np.angle(dft, 
                         deg=False).T

        data = np.array([magnitud,
                        phase])
        
        # get shape [bins, frames, channels]
        # where frames are audio frames, bins are the frequency bins
        # and channels refer to magnitud and phase.
        data = np.moveaxis(data, 0, 2)
        
        if self.transform is not None : 
            data = self.transform(data)

        return data

    # Return the number of images
    def __len__(self):
        return len(self.audio_df)