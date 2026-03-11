# EMG-to-Pose LSTM

Predicting hand joint angles from surface EMG signals using LSTM networks.

This project was developed as part of the [Erdos Institute](https://www.erdosinstitute.org/) Deep Learning program.

## Overview

This repository builds an LSTM-based pipeline for mapping 16-channel EMG wristband signals to 20 degree-of-freedom hand joint angles at 2kHz. The data loading, session handling, and preprocessing code is adapted from Meta's **emg2pose** repository:

> **emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation**
> Sasha Salter, Richard Warren, Collin Schlager, Adrian Spurr, Shangchen Han, Rohin Bhasin, Yujun Cai, Peter Walkington, Anuoluwapo Bolarinwa, Robert Wang, Nathan Danielson, Josh Merel, Eftychios Pnevmatikakis, Jesse Marshall
> [arXiv:2412.02725]([https://arxiv.org/abs/2412.02725])
> [GitHub: facebookresearch/emg2pose](https://github.com/facebookresearch/emg2pose)

The original emg2pose codebase is released under CC BY-NC-SA 4.0. Data utility functions, session loading, transforms, and alignment code in the `data/` module are extracted and modified from that repository.

## Repository Structure

```
.
├── model.py            # LSTM models (EMGPoseLSTM, SequentialEMGPoseLSTM)
├── train.py            # Training loop with checkpointing and early stopping
├── load_data.py        # Data loading pipeline and get_dataloaders()
├── data/
│   ├── session.py      # HDF5 session reader, windowed dataset, z-score scaling
│   ├── transforms.py   # EMG transforms (extraction, augmentation, downsampling)
│   ├── alignment.py    # Temporal alignment utilities
│   └── utils.py        # Split loading, IK failure masking, downsampling
├── notebooks/
│   ├── emg_pose_lstm_colab.ipynb   # Colab notebook for training
│   └── lr_search_colab.ipynb       # Colab notebook for learning rate search
└── README.md
```

## Models

- **EMGPoseLSTM** -- Processes full sequences in one shot. Input: `(B, 16, T)`, Output: `(B, 20, T)`. Used for training.
- **SequentialEMGPoseLSTM** -- Steps through one timestep at a time with persistent hidden state. Used for streaming inference.

## Data

The pipeline uses the emg2pose dataset:

- **Full dataset** (431 GiB, 25k sessions): `curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar" -o emg2pose_dataset.tar`

Each session is an HDF5 file containing synchronized 16-channel EMG and 20-DOF joint angle ground truth at 2kHz.

## Usage

### Training

```bash
# Train on real data
python train.py --data_dir /path/to/hdf5s --epochs 100

# Train with test set evaluation
python train.py --data_dir /path/to/hdf5s --epochs 100 --use_test

# Quick test with synthetic data
python train.py --test --epochs 5

# Custom hyperparameters and output directory
python train.py --data_dir /path/to/hdf5s --lr 5e-4 --hidden_size 512 --num_layers 3 --output_dir results/run1
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | auto | Path to HDF5 session files |
| `--metadata` | auto | Path to metadata CSV |
| `--test` | off | Use synthetic test data |
| `--epochs` | 100 | Maximum training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--batch_size` | 64 | Batch size |
| `--hidden_size` | 512 | LSTM hidden dimension |
| `--num_layers` | 2 | Number of LSTM layers |
| `--use_test` | off | Include test set evaluation |
| `--output_dir` | checkpoints | Where to save outputs |

### Outputs

Training saves to `--output_dir`:
- `best_model.pt` -- Best model checkpoint (by val loss)
- `loss_history.json` -- Per-epoch train/val/test losses
- `training_curves.png` -- Loss plot

## Preprocessing

- **Windowing**: 1-second windows (2000 samples) for training, 5-second windows (10000 samples) for validation/test
- **IK failure masking**: Timesteps where all joint angles are zero (inverse kinematics failures) are excluded from loss computation
- **Z-score normalization**: Per-user EMG scaling using pre-fitted StandardScaler objects from `scaler/user_scalers.pkl`
- **Augmentation**: Random channel rotation during training

## Loss

Masked MAE (L1) loss on joint angles, following the AngleMAE metric from emg2pose. The IK failure mask ensures the model is only trained on valid ground truth.

## Acknowledgments

- [emg2pose](https://github.com/facebookresearch/emg2pose) by Meta Platforms, Inc. for the dataset, data loading infrastructure, and model architecture inspiration (CC BY-NC-SA 4.0)
- [Erdos Institute](https://www.erdosinstitute.org/) Deep Learning program
