"""
Welford per-window, per-channel normalization for EMG data.

Usage in notebook:
    from welford import apply_welford_to_loaders
    apply_welford_to_loaders(loaders)
"""

import torch
from data.transforms import Compose as DataCompose


class WelfordNormalizeTensor:
    """Per-window, per-channel z-score normalization using vectorized PyTorch ops."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(data):
            data = torch.as_tensor(data)
        if data.ndim != 2:
            raise ValueError(f"Expected (time, channels), got {tuple(data.shape)}")
        x = data.to(torch.float32)
        if x.shape[0] == 0:
            raise ValueError("WelfordNormalizeTensor received an empty window")
        mean = torch.mean(x, dim=0, keepdim=True)
        var  = torch.var(x,  dim=0, keepdim=True, correction=0)
        std  = torch.sqrt(torch.clamp(var, min=0.0))
        return (x - mean) / (std + self.eps)


# Shared singleton — import this directly if needed
WELFORD_NORMALIZER = WelfordNormalizeTensor(eps=1e-8)


def append_welford_to_dataset(dataset) -> None:
    """Append Welford normalization to a single WindowedEmgDataset's transform."""
    existing = getattr(dataset, "transform", None)
    if isinstance(existing, DataCompose):
        transforms = list(existing.transforms)
    elif existing is None:
        transforms = []
    else:
        transforms = [existing]

    # Don't add twice
    if any(type(t).__name__ == "WelfordNormalizeTensor" for t in transforms):
        return

    dataset.transform = DataCompose([*transforms, WELFORD_NORMALIZER])


def apply_welford_to_loaders(loaders_dict: dict) -> None:
    """Apply Welford normalization to all session datasets in all loaders.

    Call this after get_dataloaders() with zscore=False:

        loaders = get_dataloaders(..., zscore=False)
        apply_welford_to_loaders(loaders)
    """
    patched = 0
    for loader in loaders_dict.values():
        for dataset in loader.dataset.datasets:
            append_welford_to_dataset(dataset)
            patched += 1
    print(f"Applied Welford normalization to {patched} session datasets")
