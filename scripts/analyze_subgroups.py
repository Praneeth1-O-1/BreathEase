import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from audio_model import AudioModel
from dataset import RespiratoryDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report checkpoint performance by COUGHVID metadata subgroup."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def summarize(frame):
    labels = frame["label"]
    preds = frame["prediction"]
    return {
        "samples": len(frame),
        "disease_prevalence": labels.mean(),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "pr_auc": average_precision_score(labels, frame["disease_probability"]),
    }


args = parse_args()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

audio_dir = "data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
csv_path = f"metadata/{args.split}.csv"
metadata = pd.read_csv(csv_path)
dataset = RespiratoryDataset(csv_path, audio_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

checkpoint = torch.load(args.checkpoint, map_location=device)
threshold = (
    args.threshold
    if args.threshold is not None
    else checkpoint.get("decision_threshold", 0.5)
)

model = AudioModel(freeze_backbone=True).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

disease_probs = []
with torch.no_grad():
    for waveforms, _ in loader:
        logits, _ = model(waveforms.squeeze(1).to(device))
        disease_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

metadata["disease_probability"] = disease_probs
metadata["prediction"] = (metadata["disease_probability"] >= threshold).astype(int)
metadata["audio_extension"] = metadata["audio_name"].str.extract(r"(\.[^.]+)$")[0]
metadata["cough_confidence_band"] = pd.cut(
    metadata["cough_detected"],
    bins=[0.8, 0.9, 0.95, 1.0],
    include_lowest=True,
)

print(f"Checkpoint: {args.checkpoint}")
print(f"Split: {args.split}; threshold: {threshold:.4f}")
print("\nOverall")
print(pd.DataFrame([summarize(metadata)]).to_string(index=False))

for column in [
    "status",
    "respiratory_condition",
    "audio_extension",
    "cough_confidence_band",
]:
    rows = []
    for group, frame in metadata.groupby(column, dropna=False, observed=False):
        rows.append({column: group, **summarize(frame)})
    print(f"\nBy {column}")
    print(pd.DataFrame(rows).to_string(index=False))

if args.output:
    metadata.to_csv(args.output, index=False)
    print(f"\nPer-sample predictions written to {args.output}")
