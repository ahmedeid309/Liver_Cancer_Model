"""src/train.py  – checkpoint-aware training helpers
----------------------------------------------------
✓ Saves **one checkpoint per epoch** in <export_dir>/<model_name>_epochXX.pt
✓ Can continue training from any epoch (start_epoch argument)
✓ Still supports optional mixed-precision (USE_AMP switch)
"""
import os, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from contextlib import nullcontext

# ---------- configurable ----------------------------------------------------
USE_AMP  = True                     # flip to False if AMP causes problems
AMP_CTX  = (lambda: torch.amp.autocast(device_type='cuda')) if USE_AMP else nullcontext
Scaler   = torch.amp.GradScaler     if USE_AMP else None
# ---------------------------------------------------------------------------


def _step(model, loss_fn, opt, scaler, x, y):
    """One optimisation step (FP32 or AMP)."""
    if scaler is None:                       # full FP32
        pred  = model(x)
        loss  = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    # mixed precision
    with AMP_CTX():
        pred  = model(x)
        loss  = loss_fn(pred, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


# --------------------------------------------------------------------------- #
#                               SEGMENTATION                                  #
# --------------------------------------------------------------------------- #
def train_segmentation(model, loader, device,
                       export_dir: str,
                       start_epoch: int = 0,
                       epochs: int = 10):
    """
    Train binary-mask U-Net.

    Parameters
    ----------
    export_dir  : directory where checkpoints are written
    start_epoch : 0 if training from scratch; otherwise last finished epoch
    epochs      : *total* epochs you want after this run ends
    """
    os.makedirs(export_dir, exist_ok=True)

    model.to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()         # safe for AMP
    scaler  = Scaler() if USE_AMP else None

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        pbar = tqdm(loader, leave=False, smoothing=0)
        for i, (x_cpu, y_cpu) in enumerate(pbar):
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            running += _step(model, loss_fn, opt, scaler, x, y)
            pbar.set_description(f"Seg E{epoch+1} Loss {running/(i+1):.4f}")


        # ------ checkpoint --------------------------------------------------
        ck_name = f"unet2d_epoch{epoch+1:02d}.pt"
        torch.save({'epoch': epoch+1,
                    'model_state': model.state_dict()}, 
                   os.path.join(export_dir, ck_name))
        # --------------------------------------------------------------------
        print(f"[Seg] Epoch {epoch+1}/{epochs} | Mean Loss: {running/len(loader):.4f}")


# --------------------------------------------------------------------------- #
#                              CLASSIFICATION                                 #
# --------------------------------------------------------------------------- #
def train_classification(model, loader, device,
                         export_dir: str,
                         start_epoch: int = 0,
                         epochs: int = 10):
    """Identical logic for multi-class classifier."""
    os.makedirs(export_dir, exist_ok=True)

    model.to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler  = Scaler() if USE_AMP else None

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        pbar = tqdm(loader, leave=False, smoothing=0)
        for x_cpu, y_cpu in pbar:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            running += _step(model, loss_fn, opt, scaler, x, y)
            pbar.set_description(f"Cls E{epoch+1} Loss {running/len(loader):.4f}")

        ck_name = f"classifier2d_epoch{epoch+1:02d}.pt"
        torch.save({'epoch': epoch+1,
                    'model_state': model.state_dict()},
                   os.path.join(export_dir, ck_name))
        print(f"[Cls] Epoch {epoch+1}/{epochs} | Mean Loss: {running/len(loader):.4f}")
