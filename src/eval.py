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
    dices = []
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            pred = model(x)
            dices.append(dice_score(pred, y).item())
    print(f"[Validation] Mean Dice Score: {np.mean(dices):.4f}")

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

    print(f"[Validation] Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")
