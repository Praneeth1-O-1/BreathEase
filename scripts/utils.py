import torch


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_f1,
    path,
    decision_threshold=0.5,
):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_f1": best_val_f1,
        "decision_threshold": decision_threshold,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return (
        checkpoint["epoch"],
        checkpoint.get("best_val_f1", checkpoint.get("best_val_acc"))
    )
