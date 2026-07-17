import torch
from torch.utils.data import DataLoader

from dataset import RespiratoryDataset
from audio_model import AudioModel

dataset = RespiratoryDataset(
    csv_file="metadata/train.csv",
    audio_dir="data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
)

loader = DataLoader(dataset, batch_size=2)

waveforms, labels = next(iter(loader))
waveforms = waveforms.squeeze(1)

model = AudioModel(
    checkpoint_path="Cnn14_mAP=0.431.pth"
)

model.eval()

with torch.no_grad():
    logits, weights = model(waveforms)

print("Logits :", logits.shape)
print("Weights :", weights.shape)