import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)


def evaluate(model, loader, criterion, device, optimize_threshold=False):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []
    all_disease_probs = []

    # Print only the first 20 validation samples
    printed = 0

    with torch.no_grad():

        for waveforms, labels in loader:

            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            logits, _ = model(waveforms)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            # ----------------------------
            # DEBUG: Print first 20 samples
            # ----------------------------
            if printed < 20:

                for i in range(labels.size(0)):

                    print(
                        f"GT={labels[i].item()} | "
                        f"P(Healthy)={probs[i][0].item():.4f} | "
                        f"P(Disease)={probs[i][1].item():.4f} | "
                        f"Pred={preds[i].item()}"
                    )

                    printed += 1

                    if printed == 20:
                        break

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_disease_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    disease_probs_np = np.asarray(all_disease_probs)

    acc = accuracy_score(labels_np, preds_np)

    # Unlike ordinary accuracy, this gives equal importance to Healthy and
    # Disease, so an all-Healthy classifier cannot look deceptively strong.
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)

    precision = precision_score(
        labels_np,
        preds_np,
        zero_division=0
    )

    recall = recall_score(
        labels_np,
        preds_np,
        zero_division=0
    )

    f1 = f1_score(
        labels_np,
        preds_np,
        zero_division=0
    )

    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    disease_prediction_rate = preds_np.mean()

    roc_auc = roc_auc_score(labels_np, disease_probs_np)
    pr_auc = average_precision_score(labels_np, disease_probs_np)

    best_threshold = None
    best_threshold_f1 = None
    if optimize_threshold:
        precision_curve, recall_curve, thresholds = precision_recall_curve(
            labels_np,
            disease_probs_np,
        )
        threshold_f1 = (
            2 * precision_curve[:-1] * recall_curve[:-1]
            / (precision_curve[:-1] + recall_curve[:-1] + 1e-12)
        )
        threshold_index = int(np.argmax(threshold_f1))
        best_threshold = float(thresholds[threshold_index])
        best_threshold_f1 = float(threshold_f1[threshold_index])

    return (
        avg_loss,
        acc,
        balanced_acc,
        precision,
        recall,
        f1,
        specificity,
        roc_auc,
        pr_auc,
        disease_prediction_rate,
        best_threshold,
        best_threshold_f1,
        cm
    )
