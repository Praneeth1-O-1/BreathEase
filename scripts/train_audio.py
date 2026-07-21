import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RespiratoryDataset
from audio_model import AudioModel
from evaluate import evaluate
from utils import save_checkpoint


# ---------------------------------------------------
# Device
# ---------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"\nUsing device : {device}")


# ---------------------------------------------------
# Kaggle Detection
# ---------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:

    AUDIO_DIR = (
        "/kaggle/input/datasets/pr4neeth/"
        "coughvid/coughvid_v3/"
        "coughvid_v3/"
        "public_dataset_v3/"
        "coughvid_20211012"
    )

    CHECKPOINT_PATH = (
        "/kaggle/input/datasets/pr4neeth/"
        "cnn14-pth/"
        "Cnn14_mAP0.431.pth"
    )

else:

    AUDIO_DIR = (
        "data/raw/coughvid_v3/"
        "public_dataset_v3/"
        "coughvid_20211012"
    )

    CHECKPOINT_PATH = "Cnn14_mAP=0.431.pth"


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
train_dataset = RespiratoryDataset(
    csv_file="metadata/train.csv",
    audio_dir=AUDIO_DIR
)

val_dataset = RespiratoryDataset(
    csv_file="metadata/val.csv",
    audio_dir=AUDIO_DIR
)


print(f"Train Samples : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")


# ---------------------------------------------------
# DataLoader
# ---------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------
model = AudioModel(
    checkpoint_path=CHECKPOINT_PATH,
    freeze_backbone=True
)

model = model.to(device)


# ---------------------------------------------------
# Loss
# ---------------------------------------------------
class_counts = torch.tensor(
    [6230, 3244],
    dtype=torch.float32
)

weights = class_counts.sum() / (2 * class_counts)
weights = weights.to(device)

print("Class Weights:", weights)

criterion = nn.CrossEntropyLoss(
    weight=weights
)
# ---------------------------------------------------
# Optimizer
# ---------------------------------------------------
head_parameters = (
    list(model.projection.parameters()) +
    list(model.residual.parameters()) +
    list(model.classifier.parameters())
)

optimizer = torch.optim.AdamW(
    [{"params": head_parameters, "lr": 3e-4}],
    weight_decay=1e-4
)


# ---------------------------------------------------
# Training
# ---------------------------------------------------
num_epochs = 20
# Do not save a checkpoint that never predicts the Disease class.
best_f1 = 0.0

print("\nStarting Training...\n")

for epoch in range(num_epochs):

    model.train()
    # A frozen feature extractor must also be in eval mode: this disables its
    # SpecAugment and dropout, so train and validation receive identical CNN14
    # features. The classification head remains in train mode.
    model.backbone.eval()

    train_preds = []
    train_labels = []

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (waveforms, labels) in enumerate(train_loader):

        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(waveforms)

        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        # This returns the total norm *before* clipping.  Log it to determine
        # whether global clipping is suppressing the classifier gradients.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        if batch_idx % 20 == 0:
            print(f"Pre-clip gradient norm: {grad_norm.item():.2f}")

        optimizer.step()

        if batch_idx == 0:
            print("\nClassifier Weight Mean:",
                model.classifier.weight.mean().item())

            print("Classifier Bias:",
                model.classifier.bias.detach().cpu().numpy())

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / len(train_loader)
    train_acc = running_correct / total_samples

    from sklearn.metrics import confusion_matrix

    train_cm = confusion_matrix(
        train_labels,
        train_preds
    )

    print("\nTraining Confusion Matrix")
    print(train_cm)

    print("\nTraining Finished")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------
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
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

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
    print(
        f"Best validation F1 threshold : {best_threshold:.4f} "
        f"(F1={best_threshold_f1:.4f})"
    )

    print("\nConfusion Matrix")
    print(cm)

    # ---------------------------------------------------
    # Save Best Model
    # ---------------------------------------------------
    # Accuracy alone rewards the 66% Healthy majority baseline.  Disease F1
    # reflects the clinical class we need the checkpoint to retain.
    if best_threshold_f1 > best_f1:

        best_f1 = best_threshold_f1

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            "audio_model_stage1.pth",
            decision_threshold=best_threshold,
        )

        print("\nBest model saved.")
