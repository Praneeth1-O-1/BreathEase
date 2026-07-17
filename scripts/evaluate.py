import torch

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

    with torch.no_grad():

        for waveforms, labels in loader:

            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            logits, _ = model(waveforms)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

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