"""Memmap-backed Dataset for the binarized chess data."""

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import torch
import numpy as np


class ChessPositionDataset(Dataset):
    """One row = one (position, move) training example."""

    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())

        # mmap_mode="r" -> read-only, no copy until accessed.
        self.tokens    = np.load(d / "tokens.npy",    mmap_mode="r")
        self.from_sq   = np.load(d / "from_sq.npy",   mmap_mode="r")
        self.to_sq     = np.load(d / "to_sq.npy",     mmap_mode="r")
        self.promotion = np.load(d / "promotion.npy", mmap_mode="r")

        self.n = self.meta["n_positions"]
        # seq_len lives in the parent meta.json now, not the per-split one.
        # Skip the shape assert if it's missing; the embedding layer will
        # catch any real mismatch immediately.
        if "seq_len" in self.meta:
            assert self.tokens.shape == (self.n, self.meta["seq_len"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        # .copy() because the memmap slice is read-only; torch warns otherwise.
        # uint8 -> long needed for embedding lookup.
        return {
            "tokens":    torch.from_numpy(self.tokens[idx].copy()).long(),
            "from_sq":   int(self.from_sq[idx]),
            "to_sq":     int(self.to_sq[idx]),
            "promotion": int(self.promotion[idx]),
        }


def make_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = False
):
    """Build a DataLoader for one split ("train" or "val").

    Loads from `data_dir/<split>/`. Train shuffles and drops the last
    partial batch; val iterates in order and keeps every example so
    eval metrics cover the full set.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    is_train = split == "train"
    ds = ChessPositionDataset(Path(data_dir) / split)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train or shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )