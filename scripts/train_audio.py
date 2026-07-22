import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from audio_model import AudioModel
from dataset import RespiratoryDataset
from evaluate import evaluate
from utils import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--unfreeze-final-block", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_validation_metrics(metrics, include_threshold=False):
    (
        val_loss,
        val_acc,
        val_balanced_acc,
        precision,
        recall,
        f1,
        specificity,
        roc_auc,
        pr_auc,
        disease_prediction_rate,
        best_threshold,
        best_threshold_f1,
        cm,
    ) = metrics

    print("\nValidation Results")
    print("-------------------------")
    print(f"Loss      : {val_loss:.4f}")
    print(f"Accuracy  : {val_acc:.4f}")
    print(f"Balanced Accuracy : {val_balanced_acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")
    print(f"PR-AUC      : {pr_auc:.4f}")
    print(f"Disease prediction rate : {disease_prediction_rate:.4f}")
    if include_threshold:
        print(
            f"Best validation F1 threshold : {best_threshold:.4f} "
            f"(F1={best_threshold_f1:.4f})"
        )
    print("\nConfusion Matrix")
    print(cm)


args = parse_args()
set_seed(args.seed)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\nUsing device : {device}")
print(f"Random seed : {args.seed}")

IS_KAGGLE = os.path.exists("/kaggle")
if IS_KAGGLE:
    AUDIO_DIR = (
        "/kaggle/input/datasets/pr4neeth/coughvid/coughvid_v3/"
        "coughvid_v3/public_dataset_v3/coughvid_20211012"
    )
    CHECKPOINT_PATH = "/kaggle/input/datasets/pr4neeth/cnn14-pth/Cnn14_mAP0.431.pth"
else:
    AUDIO_DIR = "data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"
    CHECKPOINT_PATH = "Cnn14_mAP=0.431.pth"

train_dataset = RespiratoryDataset("metadata/train.csv", AUDIO_DIR)
val_dataset = RespiratoryDataset("metadata/val.csv", AUDIO_DIR)
print(f"Train Samples : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
)

model = AudioModel(
    checkpoint_path=CHECKPOINT_PATH,
    freeze_backbone=True,
    unfreeze_final_block=args.unfreeze_final_block,
).to(device)

class_counts = torch.tensor([6230, 3244], dtype=torch.float32)
weights = (class_counts.sum() / (2 * class_counts)).to(device)
print("Class Weights:", weights)
criterion = nn.CrossEntropyLoss(weight=weights)

head_parameters = (
    list(model.projection.parameters())
    + list(model.residual.parameters())
    + list(model.classifier.parameters())
)
optimizer_groups = [{"params": head_parameters, "lr": args.head_lr}]
if args.unfreeze_final_block:
    optimizer_groups.append(
        {"params": model.backbone.conv_block6.parameters(), "lr": args.backbone_lr}
    )
optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)

mode = "final_block" if args.unfreeze_final_block else "frozen"
checkpoint_path = f"audio_model_{mode}_seed{args.seed}.pth"
best_pr_auc = float("-inf")
best_epoch = None
epochs_without_improvement = 0

print("\nStarting Training...\n")
for epoch in range(args.epochs):
    model.train()
    # Keep BatchNorm running statistics, SpecAugment, and pretrained dropout
    # fixed for both frozen and final-block fine-tuning experiments.
    model.backbone.eval()

    running_loss = 0.0
    train_preds = []
    train_labels = []

    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if batch_idx % 20 == 0:
            print(f"Pre-clip gradient norm: {grad_norm.item():.2f}")

        optimizer.step()

        running_loss += loss.item()
        train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    print("\nTraining Confusion Matrix")
    print(confusion_matrix(train_labels, train_preds, labels=[0, 1]))
    print(f"Train Loss : {running_loss / len(train_loader):.4f}")

    # Threshold tuning is deliberately disabled during checkpoint selection.
    metrics = evaluate(model, val_loader, criterion, device, optimize_threshold=False)
    print_validation_metrics(metrics)
    pr_auc = metrics[8]

    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        best_epoch = epoch
        epochs_without_improvement = 0
        save_checkpoint(model, optimizer, epoch, best_pr_auc, checkpoint_path)
        print(f"\nBest PR-AUC checkpoint saved: {best_pr_auc:.4f}")
    else:
        epochs_without_improvement += 1
        print(
            f"PR-AUC did not improve for {epochs_without_improvement}/"
            f"{args.patience} epoch(s)."
        )
        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

print(f"\nLoading best checkpoint from epoch {best_epoch + 1}.")
load_checkpoint(model, optimizer, checkpoint_path, device)
final_metrics = evaluate(model, val_loader, criterion, device, optimize_threshold=True)
print("\nBest-checkpoint validation with one-time threshold optimization")
print_validation_metrics(final_metrics, include_threshold=True)

best_threshold = final_metrics[10]
save_checkpoint(
    model,
    optimizer,
    best_epoch,
    best_pr_auc,
    checkpoint_path,
    decision_threshold=best_threshold,
)
print(f"\nSaved {checkpoint_path} with decision threshold {best_threshold:.4f}.")
