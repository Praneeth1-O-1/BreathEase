from dataset import RespiratoryDataset

dataset = RespiratoryDataset(
    csv_file="metadata/train.csv",
    audio_dir="data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
)

print("Dataset size:", len(dataset))

waveform, label = dataset[0]

print("Waveform shape:", waveform.shape)
print("Label:", label)