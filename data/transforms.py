# Original source: Meta Platforms, Inc. (emg2pose)
# Licensed under CC BY-NC-SA 4.0
# Modified: standalone module, removed emg2pose package dependency


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ExtractToTensor:
	"""Extract a field from a numpy structured array and convert to torch.Tensor.

	Following TNC convention, the returned tensor is of shape
	(time, field/batch, electrode_channel).

	Args:
		field: Name of the field to extract (default 'emg').
	"""

	field: str = "emg"

	def __call__(self, data: np.ndarray) -> torch.Tensor:
		return torch.as_tensor(data[self.field])


@dataclass
class RotationAugmentation:
	"""Rotate EMG along the channel dimension by a random integer (-1, 0, or 1).

	Use during training for data augmentation.
	"""

	def __call__(self, data: torch.Tensor) -> torch.Tensor:
		rotation = np.random.choice([-1, 0, 1])
		return torch.roll(data, rotation, dims=-1)


@dataclass
class ChannelDownsampling:
	"""Downsample number of EMG channels by a fixed factor.

	Args:
		downsampling: Keep every Nth channel (default 2, i.e. 16 -> 8).
	"""

	downsampling: int = 2

	def __call__(self, data: torch.Tensor) -> torch.Tensor:
		return data[:, :: self.downsampling]


@dataclass
class Compose:
	"""Compose a chain of transforms to apply sequentially.

	Args:
		transforms: Sequence of callable transforms.
	"""

	transforms: Sequence[Transform[Any, Any]]

	def __call__(self, data: Any) -> Any:
		for transform in self.transforms:
			data = transform(data)
		return data
