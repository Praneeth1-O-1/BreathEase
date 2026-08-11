"""
Evaluation script for the Chest X-ray classification pipeline.

Implements:
    Step 3 — Per-class AUROC, Mean AUROC (primary metric),
              Macro F1 (with per-class threshold tuning on val set),
              Hamming loss.
    Step 7 — Automatic detection of trained backbone architecture from checkpoint
    Step 8 — AMP mixed-precision inference

Critical design:
    Thresholds are tuned on the VALIDATION set, then applied to the
    TEST set. The test set is never used for any tuning decision.

Run:
    python evaluate.py
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from tqdm import tqdm

import config
from dataset import ChestXrayDataset
from model import CXRModel


# ============================================================
# Inference (with AMP support)
# ============================================================

def run_inference(model, loader, criterion, device):
    """
    Run model inference over a DataLoader using AMP if enabled.
    """
    model.eval()
    running_loss = 0.0
    all_labels   = []
    all_probs    = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, labels)
                probs   = torch.sigmoid(outputs)

            running_loss += loss.item()
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    return running_loss / len(loader), all_labels, all_probs


# ============================================================
# Threshold Tuning
# ============================================================

def tune_thresholds(val_labels, val_probs, classes, n_steps=17):
    """
    Find per-class probability thresholds that maximize F1 on the validation set.
    """
    candidates = np.linspace(0.05, 0.90, n_steps).tolist()
    thresholds = []

    for i, cls in enumerate(classes):
        y_true = val_labels[:, i]

        if len(np.unique(y_true)) < 2:
            thresholds.append(0.5)
            continue

        best_thresh = 0.5
        best_f1     = -1.0

        for thresh in candidates:
            y_pred = (val_probs[:, i] >= thresh).astype(int)
            score  = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1     = score
                best_thresh = thresh

        thresholds.append(best_thresh)

    return thresholds


# ============================================================
# Results Reporting
# ============================================================

def print_results(test_loss, test_labels, test_probs, thresholds, classes):
    """
    Print full evaluation report.
    """
    test_preds = np.stack(
        [(test_probs[:, i] >= thresholds[i]).astype(int) for i in range(len(classes))],
        axis=1,
    )

    print("\n" + "=" * 60)
    print(f"  {'Class':<24} {'AUROC':>6}  {'Threshold':>9}")
    print("=" * 60)

    auroc_values = []
    for i, cls in enumerate(classes):
        y_true = test_labels[:, i]
        y_prob = test_probs[:, i]

        if len(np.unique(y_true)) < 2:
            print(f"  {cls:<24} {'N/A':>6}  {thresholds[i]:>9.2f}  [skipped — single class]")
            continue

        auc = roc_auc_score(y_true, y_prob)
        auroc_values.append(auc)
        print(f"  {cls:<24} {auc:>6.4f}  {thresholds[i]:>9.2f}")

    mean_auroc = float(np.mean(auroc_values)) if auroc_values else 0.0
    print("=" * 60)
    print(f"  {'Mean AUROC':<24} {mean_auroc:>6.4f}  ← primary metric")

    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    h_loss   = hamming_loss(test_labels, test_preds)

    print(f"\n  Test Loss    : {test_loss:.4f}")
    print(f"  Mean AUROC   : {mean_auroc:.4f}   ← compare to CheXNet: 0.841")
    print(f"  Macro F1     : {macro_f1:.4f}   (val-tuned thresholds)")
    print(f"  Hamming Loss : {h_loss:.4f}   (fraction of wrong labels, lower is better)")
    print()

    return mean_auroc, macro_f1, h_loss


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[eval] Device : {device}")

    # ── Load pre-split CSVs ───────────────────────────────────────────────────
    val_path  = os.path.join(config.SPLIT_DIR, "val.csv")
    test_path = os.path.join(config.SPLIT_DIR, "test.csv")

    for p in [val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Split file not found: '{p}'\n"
                "Run  python split.py  before evaluating."
            )

    val_df  = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"[eval] Val  samples : {len(val_df)}")
    print(f"[eval] Test samples : {len(test_df)}")

    # ── Checkpoint Loading & Architecture Auto-Detection ─────────────────────
    best_ckpt = os.path.join(config.OUTPUT_DIR, "checkpoints", "best_model.pth")

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at '{best_ckpt}'\n"
            "Run  python train.py  before evaluating."
        )

    checkpoint = torch.load(best_ckpt, map_location=device)
    backbone_name = checkpoint.get("backbone", config.BACKBONE)
    saved_epoch   = checkpoint.get("epoch", "?")
    saved_auroc   = checkpoint.get("val_auroc", 0.0)

    # Resolution setup for the saved model
    if backbone_name == "efficientnet_b4":
        img_size, res_size = 320, 352
    else:
        img_size, res_size = 224, 256

    print(
        f"[eval] Loaded checkpoint: epoch {saved_epoch}, "
        f"backbone = '{backbone_name}', val AUROC = {saved_auroc:.4f}"
    )

    # ── Eval transform ────────────────────────────────────────────────────────
    eval_transform = transforms.Compose([
        transforms.Resize(res_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    val_dataset  = ChestXrayDataset(val_df,  config.IMAGE_DIRS, transform=eval_transform)
    test_dataset = ChestXrayDataset(test_df, config.IMAGE_DIRS, transform=eval_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )

    # ── Instantiate Model matching Checkpoint Architecture ───────────────────
    model = CXRModel(
        backbone_name=backbone_name, num_classes=config.NUM_CLASSES, pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()

    # ── Step 1: Threshold tuning on VALIDATION set ───────────────────────────
    print("\n[eval] Running validation inference (threshold tuning)...")
    _, val_labels, val_probs = run_inference(model, val_loader, criterion, device)

    thresholds = tune_thresholds(val_labels, val_probs, config.CLASSES)

    # ── Step 2: Final evaluation on TEST set ─────────────────────────────────
    print("[eval] Running test inference (final evaluation)...")
    test_loss, test_labels, test_probs = run_inference(
        model, test_loader, criterion, device
    )

    # ── Print full results ────────────────────────────────────────────────────
    print_results(
        test_loss, test_labels, test_probs, thresholds, config.CLASSES
    )


if __name__ == "__main__":
    main()