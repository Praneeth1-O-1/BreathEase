import sys
sys.path.append("pytorch")
sys.path.append("scripts")

import torch
from models import Cnn14
from dataset import RespiratoryDataset
from torch.utils.data import DataLoader

# Dataset
dataset = RespiratoryDataset(
    csv_file="metadata/train.csv",
    audio_dir="data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)

waveforms, labels = next(iter(loader))

# CNN14 expects (batch_size, samples)
waveforms = waveforms.squeeze(1)

# Model
model = Cnn14(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527
)

checkpoint = torch.load(
    "Cnn14_mAP=0.431.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model"])
model.eval()

with torch.no_grad():
    output = model(waveforms)

embedding = output["embedding"]

print("Waveform batch:", waveforms.shape)
print("Embedding shape:", embedding.shape)