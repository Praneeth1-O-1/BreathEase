import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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


def set_batchnorm_eval(module):
    """Keep pretrained BatchNorm running statistics stable for tiny batches.

    Calling ``model.train()`` would otherwise update every CNN14 BatchNorm
    buffer from batches of four samples.  The affine BatchNorm parameters still
    receive gradients; only train-time batch statistics are disabled.
    """
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


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


# ---------------------------------------------------
# Small subset for testing
# ---------------------------------------------------
train_dataset = Subset(train_dataset, range(2000))
val_dataset = Subset(val_dataset, range(500))

print(f"Train Samples : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")


# ---------------------------------------------------
# DataLoader
# ---------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------
model = AudioModel(
    checkpoint_path=CHECKPOINT_PATH
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
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": 1e-5
        },
        {
            "params": (
                list(model.projection.parameters()) +
                list(model.residual.parameters()) +
                list(model.classifier.parameters())
            ),
            "lr": 3e-4
        }
    ],
    weight_decay=1e-4
)


# ---------------------------------------------------
# Training
# ---------------------------------------------------
num_epochs = 20
best_f1 = -1.0

print("\nStarting Training...\n")

for epoch in range(num_epochs):

    model.train()
    set_batchnorm_eval(model.backbone)

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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

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
    val_loss, val_acc, val_balanced_acc, precision, recall, f1, cm = evaluate(
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

    print("\nConfusion Matrix")
    print(cm)

    # ---------------------------------------------------
    # Save Best Model
    # ---------------------------------------------------
    # Accuracy alone rewards the 66% Healthy majority baseline.  Disease F1
    # reflects the clinical class we need the checkpoint to retain.
    if f1 > best_f1:

        best_f1 = f1

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            "audio_model_stage1.pth"
        )

        print("\nBest model saved.")
