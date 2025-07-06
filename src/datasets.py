"""
Lazy-loading 2‑D slice datasets for liver CT segmentation & classification.
Always returns tensors with shape **[1 × H × W]**, regardless of how many
components the original NIfTI slice has.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# ────────────────────────────────────────────────────────────────────────────
# helper
# ────────────────────────────────────────────────────────────────────────────

def _resolve_path(raw_path: str, root: str) -> str:
    """Return an absolute path, probing for .nii.gz if needed."""
    cand = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
    if os.path.exists(cand):
        return cand
    gz = cand if cand.endswith(".nii.gz") else f"{cand}.gz"
    if os.path.exists(gz):
        return gz
    raise FileNotFoundError(cand)


def _load_slice(path: str, z: int) -> np.ndarray:
    """Load one axial slice and reduce to 2‑D array (H×W).

    Handles images with extra dimensions like (H,W,1,3) or (H,W,3).
    The rule: after squeezing singleton axes, if more than 2 dims remain,
    keep the first channel.
    """
    arr = nib.load(path).dataobj[:, :, z]            # lazy slice
    arr = np.asarray(arr, dtype=np.float32)          # force array
    arr = np.squeeze(arr)                            # drop dims==1
    if arr.ndim > 2:                                 # e.g. RGB or vector
        arr = arr[..., 0]                            # take first component
            # Print only for masks (filename usually contains 'mask')
   
    
        

    return arr

# ────────────────────────────────────────────────────────────────────────────
# segmentation
# ────────────────────────────────────────────────────────────────────────────

class CT2DSliceSegDataset(Dataset):
    """Lazy segmentation dataset.

    Returns: `(ct_slice, mask_slice)`  → 2 tensors of shape `[1, H, W]`.
    Ensures slice index is never out‑of‑range even if CT and mask volumes
    have different depths.
    """

    def __init__(self, df, ct_root: str, mask_root: str):
        self.samples: List[Tuple[str, str, int]] = []
        for _, row in df.iterrows():
            ct_path   = _resolve_path(row["ct_path"],   ct_root)
            mask_path = _resolve_path(row["mask_path"], mask_root)

            ct_depth   = nib.load(ct_path  ).shape[2]
            mask_depth = nib.load(mask_path).shape[2]
            depth = min(ct_depth, mask_depth)  # safest common depth

            for z in range(depth):
                self.samples.append((ct_path, mask_path, z))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        ct_path, mask_path, z = self.samples[idx]
        ct   = _load_slice(ct_path,   z)
        mask = _load_slice(mask_path, z)
        return (
            torch.from_numpy(ct).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )



# ────────────────────────────────────────────────────────────────────────────
# classification
# ────────────────────────────────────────────────────────────────────────────

class CT2DSliceClassDataset(Dataset):
    """Lazy classification dataset – returns `(ct_slice, label)`."""

    def __init__(self, df, label_encoder, ct_root: str, mask_root: str):
        self.samples: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            ct_path = _resolve_path(row["ct_path"], ct_root)
            label   = int(row["cancer_type"])
            depth   = nib.load(ct_path).shape[2]
            for z in range(depth):
                self.samples.append((ct_path, label, z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ct_path, label, z = self.samples[idx]
        ct = _load_slice(ct_path, z)
        return (torch.from_numpy(ct).unsqueeze(0),
                torch.tensor(label, dtype=torch.long))
