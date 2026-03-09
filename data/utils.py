# Original source: Meta Platforms, Inc. (emg2pose)
# Licensed under CC BY-NC-SA 4.0
# Modified: extracted data utility functions only, removed hydra/training dependencies


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_splits(
	metadata_file: pd.DataFrame,
	subsample: float = 1.0,
	random_seed: int = 0,
) -> dict[str, list[str]]:
	"""Load train, val, and test dataset splits from metadata CSV.

	Args:
		metadata_file: pandas dataframe.
		subsample: Fraction of each split to sample (default 1.0 = all).
		random_seed: Random seed for reproducible subsampling.

	Returns:
		Dict mapping split names ('train', 'val', 'test') to lists of
		HDF5 filenames.
	"""
	df = metadata_file.groupby("split").apply(
		lambda x: x.sample(frac=subsample, random_state=random_seed)
	)
	df.reset_index(drop=True, inplace=True)

	splits = {}
	for split, df_ in df.groupby("split"):
		splits[split] = list(df_.filename)

	return splits


def get_contiguous_ones(binary_vector: np.ndarray) -> list[tuple[int, int]]:
	"""Get (start_idx, end_idx) pairs for each contiguous block of True values.

	Args:
		binary_vector: 1-D boolean array.

	Returns:
		List of (start, end) index tuples for each contiguous True block.
	"""
	if (binary_vector == 0).all():
		return []

	ones = np.where(binary_vector)[0]
	boundaries = np.where(np.diff(ones) != 1)[0]
	return [
		(ones[i], ones[j])
		for i, j in zip(
			np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1)
		)
	]


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
	"""Compute mask that is True where there are no inverse kinematics failures.

	IK failure is detected when all joint angles at a timestep are zero.

	Args:
		joint_angles: Array of shape (..., n_joints).

	Returns:
		Boolean array of shape (...,) where True = valid sample.
	"""
	zeros = np.zeros_like(joint_angles)
	is_zero = np.isclose(joint_angles, zeros)
	return ~np.all(is_zero, axis=-1)


def downsample(
	array: np.ndarray, native_fs: int = 2000, target_fs: int = 30
) -> np.ndarray:
	"""Downsample array from native sampling frequency to target frequency.

	Uses linear interpolation to resample the signal.

	Args:
		array: Input array with time along axis 0.
		native_fs: Native sampling rate in Hz (default 2000).
		target_fs: Target sampling rate in Hz (default 30).

	Returns:
		Downsampled array.
	"""
	t_native = np.arange(array.shape[0]) / native_fs
	num_samples = int(array.shape[0] * target_fs / native_fs)
	t_target = np.linspace(0, t_native[-1], num_samples)

	f = interp1d(
		t_native, array, axis=0, kind="linear", fill_value=np.nan, bounds_error=False
	)
	return f(t_target)
