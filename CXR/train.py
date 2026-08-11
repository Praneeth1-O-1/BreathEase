"""
Training script for the Chest X-ray classification pipeline.

Implements:
    Step 1 — Patient-level splits from disk (split.py)
    Step 2 — Validation loop, early stopping, best-checkpoint saving, pos_weight
    Step 4 — Configurable image resolution & separate train/eval transforms
    Step 5 — AdamW + weight decay + layer-wise LRs + gradient clipping + two-stage fine-tuning
    Step 6 — OneCycleLR scheduler (warmup + cosine annealing, per-batch)
    Step 7 — Configurable backbone (DenseNet121 baseline or EfficientNet-B4)
    Step 8 — Automatic Mixed Precision (AMP) using torch.amp
    Step 9 — CSV & TensorBoard experiment logging, reproducible seed & checkpointing

Run:
    python train.py
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
import split as split_module
from dataset import ChestXrayDataset
from model import CXRModel


# ============================================================
# TensorBoard Logger (Optional with graceful fallback)
# ============================================================

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """Fix all random seeds for bit-exact reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# Step 4 — Transforms
# ============================================================

def build_transforms():
    """
    Return (train_transform, eval_transform) using config.RESIZE_SIZE and config.IMAGE_SIZE.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


# ============================================================
# Positive-class weights
# ============================================================

def compute_pos_weight(train_df, classes, device):
    """
    Compute per-class positive weights for BCEWithLogitsLoss from training set labels.
    """
    N          = len(train_df)
    pos_counts = np.zeros(len(classes), dtype=np.float64)

    for labels_str in train_df["Finding Labels"]:
        for disease in str(labels_str).split("|"):
            disease = disease.strip()
            if disease in classes:
                pos_counts[classes.index(disease)] += 1

    pos_counts = np.where(pos_counts == 0, 1.0, pos_counts)
    neg_counts = N - pos_counts
    weights    = neg_counts / pos_counts

    print("\n[train] Class balance (training set):")
    print(f"  {'Class':<22} {'Positives':>9} {'pos_weight':>10}")
    print(f"  {'-'*22} {'-'*9} {'-'*10}")
    for cls, cnt, w in zip(classes, pos_counts.astype(int), weights):
        print(f"  {cls:<22} {cnt:>9}  {w:>10.2f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================
# Checkpoint Helper
# ============================================================

def save_checkpoint(model, optimizer, epoch, val_auroc, val_loss, path):
    """Save full checkpoint with state dicts, hyperparams, and backbone metadata."""
    torch.save(
        {
            "epoch"               : epoch,
            "backbone"            : config.BACKBONE,
            "image_size"          : config.IMAGE_SIZE,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc"           : val_auroc,
            "val_loss"            : val_loss,
        },
        path,
    )


# ============================================================
# Step 9 — CSV Metrics Logger
# ============================================================

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Initialize file with header
        with open(self.filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "stage", "train_loss", "val_loss", "val_auroc", "lr_backbone", "lr_head"
            ])

    def log(self, epoch, stage, train_loss, val_loss, val_auroc, lr_backbone, lr_head):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, stage, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_auroc:.4f}",
                f"{lr_backbone:.2e}", f"{lr_head:.2e}"
            ])


# ============================================================
# Step 5 & 8 — Training Epoch with AMP & Gradient Clipping
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, scheduler=None, freeze_backbone_bn=False):
    """
    Run one training epoch with optional Automatic Mixed Precision (AMP).
    """
    model.train()
    if freeze_backbone_bn and hasattr(model, "features"):
        model.features.eval()

    running_loss = 0.0
    progress = tqdm(loader, desc="  Train", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Step 8: AMP autocast
        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if config.USE_AMP and scaler is not None:
            scaler.scale(loss).backward()

            if config.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


# ============================================================
# Validation Epoch (with AMP)
# ============================================================

def val_epoch(model, loader, criterion, device):
    """
    Run validation pass.
    """
    model.eval()
    running_loss = 0.0
    all_labels   = []
    all_probs    = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
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
# AUROC Helper
# ============================================================

def compute_mean_auroc(labels, probs, classes):
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for i in range(len(classes)):
        if len(np.unique(labels[:, i])) > 1:
            aurocs.append(roc_auc_score(labels[:, i], probs[:, i]))
    return float(np.mean(aurocs)) if aurocs else 0.0


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    set_seed(config.SEED)

    # ── Output directories ───────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")

    # ── Device & AMP Scaler ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device     : {device}")
    print(f"[train] AMP (FP16) : {config.USE_AMP}")
    print(f"[train] Backbone   : {config.BACKBONE} (image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE})")

    # Step 8: AMP GradScaler
    scaler = torch.amp.GradScaler('cuda') if (config.USE_AMP and device.type == "cuda") else None

    # ── Step 9: Setup CSV and TensorBoard Loggers ────────────────────────────
    csv_logger = CSVLogger(config.CSV_LOG_FILE)
    tb_writer  = SummaryWriter(config.TB_LOG_DIR) if HAS_TENSORBOARD else None

    if HAS_TENSORBOARD:
        print(f"[train] TensorBoard : {config.TB_LOG_DIR}")

    # ── Splits ───────────────────────────────────────────────────────────────
    train_df, val_df, _ = split_module.create_splits()
    print(f"\n[train] Train samples : {len(train_df)}")
    print(f"[train] Val   samples : {len(val_df)}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform, eval_transform = build_transforms()

    train_dataset = ChestXrayDataset(train_df, config.IMAGE_DIRS, transform=train_transform)
    val_dataset   = ChestXrayDataset(val_df,   config.IMAGE_DIRS, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )

    # ── Positive-class weights ────────────────────────────────────────────────
    pos_weight = compute_pos_weight(train_df, config.CLASSES, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CXRModel(
        backbone_name=config.BACKBONE, num_classes=config.NUM_CLASSES, pretrained=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train] {config.BACKBONE} loaded | Parameters : {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = 0.0
    no_improve = 0

    # =========================================================================
    # Stage 1 — Backbone FROZEN, train head only
    # =========================================================================

    if config.FREEZE_EPOCHS > 0:
        print(
            f"\n{'─'*65}"
            f"\n[Stage 1] Backbone FROZEN — training head only ({config.FREEZE_EPOCHS} epochs)"
            f"\n{'─'*65}"
        )

        for param in model.features.parameters():
            param.requires_grad = False
        model.features.eval()

        optimizer_s1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = config.LR_HEAD,
            weight_decay = config.WEIGHT_DECAY,
        )

        for epoch in range(1, config.FREEZE_EPOCHS + 1):
            print(f"\n[Stage 1] Epoch {epoch}/{config.FREEZE_EPOCHS}")

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer_s1, device,
                scaler=scaler, freeze_backbone_bn=True
            )
            val_loss, val_labels, val_probs = val_epoch(model, val_loader, criterion, device)
            val_auroc = compute_mean_auroc(val_labels, val_probs, config.CLASSES)

            lr_backbone = 0.0
            lr_head     = optimizer_s1.param_groups[0]["lr"]

            print(
                f"  Train Loss : {train_loss:.4f}  |  "
                f"Val Loss : {val_loss:.4f}  |  "
                f"Val AUROC : {val_auroc:.4f}"
            )

            # Step 9: Logging
            csv_logger.log(epoch, "Stage1", train_loss, val_loss, val_auroc, lr_backbone, lr_head)
            if tb_writer:
                tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                tb_writer.add_scalar("Loss/Val",   val_loss, epoch)
                tb_writer.add_scalar("AUROC/Val",  val_auroc, epoch)

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                save_checkpoint(model, optimizer_s1, epoch, val_auroc, val_loss, best_ckpt)
                print(f"  ✓ New best Val AUROC {best_auroc:.4f} — checkpoint saved.")

        for param in model.features.parameters():
            param.requires_grad = True

        print(f"\n[Stage 1] Complete — backbone unfrozen.")

    # =========================================================================
    # Stage 2 — Full fine-tuning + OneCycleLR + Early Stopping
    # =========================================================================

    remaining_epochs = config.EPOCHS - config.FREEZE_EPOCHS

    if remaining_epochs <= 0:
        print("[train] No Stage 2 epochs remaining (FREEZE_EPOCHS == EPOCHS).")
        print(f"[train] Best Val AUROC : {best_auroc:.4f}")
        return

    print(
        f"\n{'─'*65}"
        f"\n[Stage 2] Backbone UNFROZEN — full fine-tuning "
        f"({remaining_epochs} epochs, early stopping patience={config.PATIENCE})"
        f"\n{'─'*65}"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(),   "lr": config.LR_BACKBONE},
            {"params": model.classifier.parameters(), "lr": config.LR_HEAD},
        ],
        weight_decay = config.WEIGHT_DECAY,
    )

    total_steps = remaining_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = [config.LR_BACKBONE, config.LR_HEAD],
        total_steps     = total_steps,
        pct_start       = config.PCT_START,
        anneal_strategy = "cos",
        div_factor      = 25.0,
        final_div_factor= 1e4,
    )

    no_improve = 0

    for epoch in range(config.FREEZE_EPOCHS + 1, config.EPOCHS + 1):
        print(f"\n[Stage 2] Epoch {epoch}/{config.EPOCHS}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, scheduler=scheduler, freeze_backbone_bn=False
        )
        val_loss, val_labels, val_probs = val_epoch(model, val_loader, criterion, device)
        val_auroc = compute_mean_auroc(val_labels, val_probs, config.CLASSES)

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_backbone, lr_head = current_lrs[0], current_lrs[1]

        print(
            f"  Train Loss : {train_loss:.4f}  |  "
            f"Val Loss : {val_loss:.4f}  |  "
            f"Val AUROC : {val_auroc:.4f}"
        )
        print(f"  LR backbone={lr_backbone:.2e}, head={lr_head:.2e}")

        # Step 9: Logging
        csv_logger.log(epoch, "Stage2", train_loss, val_loss, val_auroc, lr_backbone, lr_head)
        if tb_writer:
            tb_writer.add_scalar("Loss/Train",    train_loss, epoch)
            tb_writer.add_scalar("Loss/Val",      val_loss, epoch)
            tb_writer.add_scalar("AUROC/Val",     val_auroc, epoch)
            tb_writer.add_scalar("LR/Backbone",   lr_backbone, epoch)
            tb_writer.add_scalar("LR/Head",       lr_head, epoch)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_auroc, val_loss, best_ckpt)
            print(f"  ✓ New best Val AUROC {best_auroc:.4f} — checkpoint saved.")

        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{config.PATIENCE} epochs.")
            if no_improve >= config.PATIENCE:
                print(
                    f"\n[train] Early stopping after {epoch} epochs "
                    f"(no AUROC improvement for {config.PATIENCE} consecutive epochs)."
                )
                break

    if tb_writer:
        tb_writer.close()

    print(f"\n{'─'*65}")
    print(f"[train] Training complete.")
    print(f"[train] Best Val AUROC : {best_auroc:.4f}")
    print(f"[train] CSV Log        : {config.CSV_LOG_FILE}")
    print(f"[train] Checkpoint     : {best_ckpt}")
    print(f"\nNext step: run  python evaluate.py")


if __name__ == "__main__":
    main()