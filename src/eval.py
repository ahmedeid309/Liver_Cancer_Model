# src/eval.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

def dice_score(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)


def evaluate_segmentation(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    dice_scores, ious, accs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)                  # raw network output
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()     # binarise

            inter = (preds * y).sum(dim=(1,2,3))
            preds_sum = preds.sum(dim=(1,2,3))
            y_sum     = y.sum(dim=(1,2,3))

            dice = (2*inter) / (preds_sum + y_sum + 1e-8)
            iou  = inter / (preds_sum + y_sum - inter + 1e-8)
            acc  = (preds == y).float().mean(dim=(1,2,3))

            dice_scores.extend(dice.cpu().tolist())
            ious.extend(iou.cpu().tolist())
            accs.extend(acc.cpu().tolist())

    print(f"[Seg] Dice {np.mean(dice_scores):.4f} | IoU {np.mean(ious):.4f} | "
          f"PixelAcc {np.mean(accs):.4f}")


def evaluate_classification(model, dataset, label_encoder, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataset:
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            y_true.append(y)
            y_pred.append(pred.argmax(dim=1).item())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"[Classification Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")
