import os
import pandas as pd
import torch
import torchaudio

from torch.utils.data import Dataset


class RespiratoryDataset(Dataset):

    TARGET_SAMPLE_RATE = 32000

    def __init__(self, csv_file, audio_dir):

        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["audio_name"]
        label = int(row["label"])

        audio_path = os.path.join(
            self.audio_dir,
            filename
        )

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                self.TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        TARGET_LENGTH = 320000

        current_length = waveform.shape[1]

        if current_length < TARGET_LENGTH:

            pad = TARGET_LENGTH - current_length

            waveform = torch.nn.functional.pad(
                waveform,
                (0, pad)
            )

        elif current_length > TARGET_LENGTH:

            waveform = waveform[:, :TARGET_LENGTH]

        return waveform, label