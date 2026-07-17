from dataset import RespiratoryDataset

from torch.utils.data import DataLoader

dataset = RespiratoryDataset(
    csv_file="metadata/train.csv",
    audio_dir="data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

waveforms, labels = next(iter(loader))

print(waveforms.shape)

print(labels)