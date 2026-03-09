#!/usr/bin/env python3
"""
Load and align emg2pose data for model training.

Downloads instructions:
  Mini dataset (~600 MiB):
    curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar" -o emg2pose_dataset_mini.tar
    tar -xvf emg2pose_dataset_mini.tar

  Metadata CSV (5 MiB):
    curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv -o emg2pose_metadata.csv

  Full dataset (431 GiB):
    curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar
    tar -xvf emg2pose_dataset.tar

Usage:
  # Default: load mini dataset from sibling repo
  python load_data.py

  # Explicit paths
  python load_data.py --data_dir /path/to/hdf5s --metadata /path/to/metadata.csv

  # Quick test with synthetic data (no download needed)
  python load_data.py --test
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.session import (
	Emg2PoseSessionData, WindowedEmgDataset,
	load_user_scalers, DEFAULT_SCALER_PATH,
)
from data.transforms import ExtractToTensor, RotationAugmentation, Compose
from data.utils import load_splits, downsample, get_ik_failures_mask
from data.alignment import align_predictions, align_mask


# ---------------------------------------------------------------------------
# Constants from emg2pose
# ---------------------------------------------------------------------------
EMG_SAMPLE_RATE = 2000      # Hz
EMG_CHANNELS = 16
JOINT_ANGLE_CHANNELS = 20
WINDOW_LENGTH = 2000        # 1 second at 2kHz (training)
VAL_WINDOW_LENGTH = 10000   # 5 seconds at 2kHz (validation/test)
BATCH_SIZE = 64

# Default data location (mini dataset in sibling repo)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "fall-2025-gesture-recognition" / "data" / "emg2pose_dataset_mini"


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------
def create_test_hdf5(path: Path, n_samples: int = 20000) -> Path:
	"""Create a synthetic HDF5 file matching emg2pose format for testing.

	Args:
		path: Directory to write the file into.
		n_samples: Number of time samples (default 20000 = 10 sec at 2kHz).

	Returns:
		Path to the created HDF5 file.
	"""
	filepath = path / "test_session.hdf5"
	timeseries_dtype = np.dtype([
		("time", "<f8"),
		("joint_angles", "<f8", (JOINT_ANGLE_CHANNELS,)),
		("emg", "<f4", (EMG_CHANNELS,)),
	])
	ts = np.empty(n_samples, dtype=timeseries_dtype)
	ts["time"] = np.arange(n_samples, dtype="<f8") / EMG_SAMPLE_RATE
	ts["emg"] = np.random.randn(n_samples, EMG_CHANNELS).astype(np.float32)
	ts["joint_angles"] = np.random.randn(n_samples, JOINT_ANGLE_CHANNELS)

	# Insert a few IK failures (all-zero joint angles) to test masking
	ts["joint_angles"][100:110] = 0.0

	metadata = {
		"filename": "test_session",
		"session": "test_session",
		"stage": "test_stage",
		"user": "test_user",
		"side": "right",
		"sample_rate": float(EMG_SAMPLE_RATE),
		"num_channels": EMG_CHANNELS,
		"start": 0.0,
		"end": float(n_samples) / EMG_SAMPLE_RATE,
	}

	with h5py.File(filepath, "w") as f:
		group = f.create_group("emg2pose")
		group["timeseries"] = ts
		group.attrs.update(metadata)

	print(f"Created synthetic HDF5: {filepath} ({n_samples} samples, "
		  f"{n_samples / EMG_SAMPLE_RATE:.1f}s)")
	return filepath


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_datasets(
	hdf5_paths: list[Path],
	window_length: int,
	stride: int | None = None,
	jitter: bool = False,
	skip_ik_failures: bool = True,
	augment: bool = False,
	user_scalers: dict | None = None,
) -> ConcatDataset:
	"""Build a ConcatDataset from multiple HDF5 session files.

	Args:
		hdf5_paths: List of paths to session HDF5 files.
		window_length: Window size in samples.
		stride: Stride between windows (None = window_length).
		jitter: Random jitter for training variability.
		skip_ik_failures: Skip windows containing IK failures.
		augment: Apply rotation augmentation (for training).
		user_scalers: Dict mapping user IDs to sklearn StandardScaler
			objects for per-user z-score normalization. None to skip.

	Returns:
		ConcatDataset wrapping all sessions.
	"""
	if augment:
		transform = Compose([ExtractToTensor(field="emg"), RotationAugmentation()])
	else:
		transform = ExtractToTensor(field="emg")

	datasets = []
	scaled_count = 0
	unscaled_users = set()

	for path in hdf5_paths:
		# Look up per-user scaler — only apply if user is present in pkl
		scaler = None
		if user_scalers is not None:
			session = Emg2PoseSessionData(path)
			user_id = session.user
			session._file.close()
			if user_id in user_scalers:
				scaler = user_scalers[user_id]
				scaled_count += 1
			else:
				unscaled_users.add(user_id)

		ds = WindowedEmgDataset(
			hdf5_path=path,
			window_length=window_length,
			stride=stride,
			jitter=jitter,
			transform=transform,
			skip_ik_failures=skip_ik_failures,
			scaler=scaler,
		)
		datasets.append(ds)

	if user_scalers is not None:
		print(f"  Z-score scaling: {scaled_count}/{len(hdf5_paths)} sessions matched a user scaler")
		if unscaled_users:
			print(f"  Users without scalers (skipped): {sorted(unscaled_users)}")

	return ConcatDataset(datasets)


def build_dataloaders(
	train_paths: list[Path],
	val_paths: list[Path],
	test_paths: list[Path],
	batch_size: int = BATCH_SIZE,
	num_workers: int = 0,
	user_scalers: dict | None = None,
	use_test: bool = False,
) -> dict[str, DataLoader]:
	"""Build train/val/test DataLoaders.

	Args:
		train_paths: HDF5 file paths for training.
		val_paths: HDF5 file paths for validation.
		test_paths: HDF5 file paths for testing.
		batch_size: Batch size for all loaders.
		num_workers: Number of dataloader workers.
		user_scalers: Dict mapping user IDs to sklearn StandardScaler
			objects for per-user z-score normalization. None to skip.
		use_test: If True, build a test DataLoader. If False (default),
			only train and validation loaders are built.

	Returns:
		Dict with 'train', 'test' (and optionally 'val') DataLoader instances.
	"""
	train_ds = build_datasets(
		train_paths,
		window_length=WINDOW_LENGTH,
		jitter=True,
		skip_ik_failures=True,
		augment=True,
		user_scalers=user_scalers,
	)

	val_ds = build_datasets(
		val_paths,
		window_length=VAL_WINDOW_LENGTH,
		skip_ik_failures=True,
		user_scalers=user_scalers,
	)
	loaders = {
		"train": DataLoader(
			train_ds, batch_size=batch_size, shuffle=True,
			num_workers=num_workers, drop_last=True,
		),
		"val": DataLoader(
			val_ds, batch_size=batch_size, shuffle=False,
			num_workers=num_workers,
		),
	}

	if use_test:
		test_ds = build_datasets(
			test_paths,
			window_length=VAL_WINDOW_LENGTH,
			skip_ik_failures=True,
			user_scalers=user_scalers,
		)

		loaders["test"] = DataLoader(
			test_ds, batch_size=batch_size, shuffle=False,
			num_workers=num_workers,
		)

	return loaders


# ---------------------------------------------------------------------------
# Alignment demo
# ---------------------------------------------------------------------------
def demonstrate_alignment(batch: dict[str, torch.Tensor]):
	"""Show how temporal alignment works on a batch.

	When the model produces predictions at a different temporal resolution
	than the targets, align_predictions resamples to match.
	"""
	emg = batch["emg"]                    # (B, C=16, T)
	joint_angles = batch["joint_angles"]  # (B, C=20, T)
	mask = batch["no_ik_failure"]         # (B, T)
	target_time = joint_angles.shape[-1]

	# Simulate a model that outputs at 1/4 temporal resolution
	simulated_pred = torch.randn(
		emg.shape[0], JOINT_ANGLE_CHANNELS, target_time // 4
	)
	aligned_pred = align_predictions(simulated_pred, target_time)
	aligned_mask = align_mask(mask, target_time)

	print(f"\n--- Alignment Demo ---")
	print(f"  Target shape:           {joint_angles.shape}")
	print(f"  Simulated pred shape:   {simulated_pred.shape}")
	print(f"  Aligned pred shape:     {aligned_pred.shape}")
	print(f"  Original mask shape:    {mask.shape}")
	print(f"  Aligned mask shape:     {aligned_mask.shape}")
	print(f"  Valid samples in mask:  {aligned_mask.sum().item()}/{aligned_mask.numel()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_dataloaders(data_dir=None, metadata=None, test_mode=False,
                    batch_size=BATCH_SIZE, num_workers=0, zscore=True,
                    use_test=False):
	"""Build train/test (and optionally val) dataloaders. Reusable by train.py.

	Args:
		data_dir: Path to directory containing HDF5 session files.
		metadata: Path to emg2pose metadata CSV with train/val/test splits.
		test_mode: If True, use synthetic test data (no download needed).
		batch_size: Batch size for all loaders.
		num_workers: Number of dataloader workers.
		zscore: If True, apply per-user z-score normalization to EMG data
			using scalers from scaler/user_scalers.pkl.
		use_test: If True, include a test DataLoader. Default False
			(only train and test).

	Returns:
		Dict with 'train', 'test' (and optionally 'val') DataLoader instances.
	"""
	# Load per-user scalers for z-score normalization
	user_scalers = None
	if zscore and DEFAULT_SCALER_PATH.exists():
		user_scalers = load_user_scalers()
		print(f"Loaded z-score scalers for {len(user_scalers)} users")

	if test_mode:
		print("=== Running with synthetic test data ===\n")
		test_dir = Path("test_data")
		test_dir.mkdir(exist_ok=True)
		hdf5_path = create_test_hdf5(test_dir)

		paths = [hdf5_path]
		loaders = build_dataloaders(
			train_paths=paths, val_paths=paths, test_paths=paths,
			batch_size=min(batch_size, 4),
			user_scalers=user_scalers,
			use_test=use_test,
		)

	else:
		data_dir = Path(data_dir).expanduser() if data_dir else DEFAULT_DATA_DIR
		assert data_dir.exists(), f"Data directory not found: {data_dir}"

		if metadata:
			metadata = Path(metadata).expanduser()
		else:
			metadata = data_dir / "metadata.csv"
		assert metadata.exists(), f"Metadata CSV not found: {metadata}"

		print(f"=== Loading data from {data_dir} ===")
		print(f"    Metadata: {metadata}\n")

		import pandas as pd
		splits = load_splits(pd.read_csv(metadata))
		available = {p.stem for p in data_dir.glob("*.hdf5")}

		for split_name in splits:
			before = len(splits[split_name])
			splits[split_name] = [f for f in splits[split_name] if f in available]
			after = len(splits[split_name])
			if after < before:
				print(f"  {split_name}: {before} in CSV -> {after} on disk")

		has_empty = any(len(splits[s]) == 0 for s in ["train", "val", "test"])
		if has_empty:
			all_files = sorted(available)
			np.random.seed(42)
			np.random.shuffle(all_files)
			n = len(all_files)
			n_train = max(1, int(0.7 * n))
			n_val = max(1, int(0.15 * n))
			splits = {
				"train": all_files[:n_train],
				"val": all_files[n_train:n_train + n_val],
				"test": all_files[n_train + n_val:],
			}
			print(f"\n  Redistributed {n} files into 70/15/15 splits")

		def resolve_paths(filenames):
			return [data_dir / f"{name}.hdf5" for name in filenames]

		print(f"\nUsing splits: train={len(splits['train'])}, "
			  f"val={len(splits['val'])}, test={len(splits['test'])}")

		n_total = sum(len(v) for v in splits.values())
		effective_batch_size = min(batch_size, 8) if n_total <= 30 else batch_size

		loaders = build_dataloaders(
			train_paths=resolve_paths(splits["train"]),
			val_paths=resolve_paths(splits["val"]),
			test_paths=resolve_paths(splits["test"]),
			batch_size=effective_batch_size,
			num_workers=num_workers,
			user_scalers=user_scalers,
			use_test=use_test,
		)

	return loaders


def main():
	parser = argparse.ArgumentParser(description="Load and align emg2pose data")
	parser.add_argument("--data_dir", type=str, default=None,
		help="Path to directory containing HDF5 session files")
	parser.add_argument("--metadata", type=str, default=None,
		help="Path to emg2pose metadata CSV with train/val/test splits")
	parser.add_argument("--test", action="store_true",
		help="Run with synthetic test data (no download needed)")
	parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
	parser.add_argument("--num_workers", type=int, default=0)
	args = parser.parse_args()

	loaders = get_dataloaders(
		data_dir=args.data_dir,
		metadata=args.metadata,
		test_mode=args.test,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)

	# --- Print data summary ---
	print(f"\n{'='*50}")
	print(f"{'Split':<8} {'Batches':>8} {'Samples':>10}")
	print(f"{'='*50}")
	for name, loader in loaders.items():
		n_samples = len(loader.dataset)
		n_batches = len(loader)
		print(f"{name:<8} {n_batches:>8} {n_samples:>10}")

	# --- Fetch one batch and inspect ---
	train_loader = loaders["train"]
	batch = next(iter(train_loader))

	print(f"\n--- First training batch ---")
	for key, val in batch.items():
		if isinstance(val, torch.Tensor):
			print(f"  {key:<20} {str(val.shape):<25} dtype={val.dtype}")
		else:
			print(f"  {key:<20} {val}")

	# --- IK failure mask summary ---
	mask = batch["no_ik_failure"]
	pct_valid = 100 * mask.float().mean().item()
	print(f"\n  IK failure mask: {pct_valid:.1f}% valid samples in batch")

	# --- Downsampling demo ---
	emg_np = batch["emg"][0].numpy()  # (C, T) for first sample
	emg_ds = downsample(emg_np.T, native_fs=EMG_SAMPLE_RATE, target_fs=30)
	print(f"\n--- Downsampling Demo ---")
	print(f"  Original:     {emg_np.shape[1]} samples at {EMG_SAMPLE_RATE} Hz")
	print(f"  Downsampled:  {emg_ds.shape[0]} samples at 30 Hz")

	# --- Alignment demo ---
	demonstrate_alignment(batch)

	print(f"\nData loading complete. Ready for model training.")


if __name__ == "__main__":
	main()
