# Original source: Meta Platforms, Inc. (emg2pose)
# Licensed under CC BY-NC-SA 4.0
# Modified: extracted as standalone functions from BasePoseModule class


import torch
from torch.nn.functional import interpolate


def align_predictions(pred: torch.Tensor, n_time: int) -> torch.Tensor:
	"""Temporally resample predictions to match the length of targets.

	Uses linear interpolation to upsample or downsample the prediction
	tensor along the time dimension.

	Args:
		pred: Prediction tensor of shape (batch, channels, time).
		n_time: Target time dimension length.

	Returns:
		Resampled tensor of shape (batch, channels, n_time).
	"""
	return interpolate(pred, size=n_time, mode="linear")


def align_mask(mask: torch.Tensor, n_time: int) -> torch.Tensor:
	"""Temporally resample a boolean mask to match the length of targets.

	Uses nearest-neighbor interpolation to preserve boolean semantics.

	Args:
		mask: Boolean mask of shape (batch, time).
		n_time: Target time dimension length.

	Returns:
		Resampled boolean mask of shape (batch, n_time).
	"""
	mask = mask[:, None].to(torch.float32)
	aligned = interpolate(mask, size=n_time, mode="nearest")
	return aligned.squeeze(1).to(torch.bool)
