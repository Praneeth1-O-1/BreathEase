import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

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

    avg_loss = total_loss / len(loader)

    acc = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    return (
        avg_loss,
        acc,
        precision,
        recall,
        f1,
        cm
    )