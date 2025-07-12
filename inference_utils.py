"""Utilities for inference – matches slice preprocessing used during training."""

from __future__ import annotations
import numpy as np
import nibabel as nib
import torch

# -----------------------------------------------------------------------------
# Helper to collapse extra channel / singleton axes ---------------------------
# -----------------------------------------------------------------------------

def _collapse_to_3d(arr: np.ndarray) -> np.ndarray:
    """Reduce *arr* to shape (H, W, D).

    Steps (repeated until only 3 dims remain):
      • `np.squeeze` to drop all singleton axes.
      • If more than 3 dims still exist, treat the first axis whose size ≤ 4 as a
        contrast/channel axis and keep index 0 along it.
    """
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        axis = next((i for i, s in enumerate(arr.shape) if s <= 4), arr.ndim - 1)
        arr = np.take(arr, indices=0, axis=axis)  # keep first channel
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Could not collapse to 3‑D shape – got {arr.shape}")
    return arr.astype(np.float32)  # H×W×D (raw HU)


# -----------------------------------------------------------------------------
# Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume and return an array with shape D×H×W."""
    data = nib.load(path).get_fdata(dtype=np.float32)
    data = _collapse_to_3d(data)   # H×W×D
    return np.moveaxis(data, 2, 0)  # D×H×W


def slice_batch_to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert D×H×W numpy array → (D, 1, H, W) torch tensor on *device*."""
    if batch.ndim != 3:
        raise ValueError(f"Expected 3‑D array (D,H,W); got {batch.shape}")
    return torch.from_numpy(batch).unsqueeze(1).to(device)
