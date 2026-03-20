# Spatiotemporal modeling of pose estimation in wearables

Members: Brian Mullen, Dayoung Lee, Kristin Dona, & Sero Toriano Parel

Instructor: Lindsay Warrenburg & Marcos Ortiz

Program: [Erdős Institute](https://www.erdosinstitute.org/) Deep Learning Boot Camp, Spring 2026

Predicting hand joint angles from surface EMG signals using LSTM networks.

## Overview

### Scope
This project aims to build a deep learning pipeline for hand pose estimation from surface electromyography (sEMG) signals recorded by a smart wristband equipped with muscle activity sensors. We focus on the core deployment pain points of generalization to new users/poses, sensor placements, and trajectory quality. We establish a baseline from the repository and add a small, well-ablated improvement through a hierarchical learning approach. This project is packaged as a reproducible PyTorch pipeline that can be run in Google Colab. Additionally, we include deployment by publishing our trained model checkpoints and inference code to Hugging Face, along with a model card describing the evaluation splits and intended use.

### Dataset
We use the `emg2pose` dataset, which includes data from 193 users, 370 hours, 16-channel sEMG signals at 2 kHz (Salter, Warren, Schlager, et al., NeurIPS 2024). For our controlled comparison, we train on a fixed 10-user subset (26.6GB, 1584 sessions) with standardized train/val/test (70/15/15) splits and Welford per-channel Z-score normalization across all models. The `emg2pose` GitHub repository can be located at https://github.com/facebookresearch/emg2pose. 

### Stakeholders
Reality Labs Wearables team: robust sEMG decoding and interaction quality
EMG modeling team working on methodologies, model evaluation, and benchmarks
Open-source/research community: reproducible baselines and improvements on a publicly available benchmark
Individuals with disabilities and accessibility advocates: reliable, low-friction experience for a wider range of users

### Approach
We compare four architectures (LSTM, CNN, CNN+LSTM, TDS+LSTM) on the same data, same preprocessing, and same training budget (50 epochs, patience 50, L1 loss). This design isolates the contribution of feature extraction (CNN/TDS), temporal modeling (LSTM), and their combination:

### Results
We trained and evaluated all four models on the same 10-user subset for 50 epochs. The models varied substantially in parameter count and architecture. The LSTM model used 812k parameters, with hidden layers of size 256. CNN-only model used 71k parameters with 3 Conv2d layers. CNN + LSTM model used 915k parameters, combining 3 convolutional layers with a 2-layer LSTM of hidden size 256. TDS + LSTM used 2.6M parameters with a single TDS convolutional block followed by an LSTM. All models were trained with Welford normalization, L1 loss, and a batch size of 64, except the CNN-only model, which used a batch size of 16.

When tested on held-out users, models diverged significantly. LSTM generalized the best, maintaining a mean absolute error of 16.1° on new users. CNN-only performed worst, jumping to 38.8°, and we suspect that it had essentially memorized the training users rather than learning anything generalizable. CNN + LSTM and TDS + LSTM models’ MAE  is 19.4° and 24.4°, respectively, which we believe reflects the fact that their additional spatial learning capacity needs more users to train properly. 


| Model | Params | Val MAE| Test MAE | Time/epoch ** |
| :-------| :------: | :-------: | :------: | :-------: |
| LSTM | 812k | 15.2° | 16.1° | ~55s |
|CNN-only| 71k | 16.8° | 18.1° | * |
| CNN + LSTM | 915k | 16.1° | 17.9° | ~52s |
|TDS + LSTM | 2.6M | 17.2° | 18.3° | ~7.5s |

* Run time not recorded for the CNN-only model 
** Run times based on google colab G4 GPU 

### Key Performance Indicators (KPIs)
To contextualize the performance of the model, we define two naive baselines. The zero-prediction baseline assumes a predicted joint angle of 0° for every joint at every timestep, yielding an MAE of approximately 25°. The mean joint angle baseline assumes we predict the mean joint angle for each joint across the dataset, yielding an MAE of approximately 14°. We clear the zero-prediction baseline across all models, but did not beat the mean joint angle baseline. We attribute this, and the 4° gap to the baseline quoted in the paper, to be due to our models only being trained on less than 10% of the full data available. Future work should prioritize access to greater computational resources in order to train over the full dataset, matching the 158 users used in the paper. 

### Deployment
We deployed our best model as a live demo on Hugging Face Spaces at huggingface.co/spaces/dayoungl/emgpose. The app accepts an EMG2pose-format HDF5 file of wrist EMG signals as input and returns a predicted 2D hand pose animation as output. Joint angle predictions from the LSTM model are passed through the forward kinematics algorithm provided by UmeTrack, which converts the 20 predicted joint angles per timestamp into an animated hand mesh rendered as a video. 

### References
Bao, T., Zaidi, S. A. R., Xie, S. Q., Yang, P., & Zhang, Z. Q. (2021). A CNN-LSTM hybrid model for wrist kinematics estimation using surface electromyography. IEEE Transactions on Instrumentation and Measurement, 70, 1–10. https://doi.org/10.1109/TIM.2020.3031070.
Han, S., Wu, P., Zhang, Y., Liu, B., Zhang, L., Wang, Z., Si, W., Zhang, P., Cai, Y., Hodan, T., Cabezas, R., Tran, L., Akbay, M., Yu, T., Keskin, C., & Wang, R. (2022). UmeTrack: Unified multi-view end-to-end hand tracking for VR. SIGGRAPH Asia 2022 Conference Papers. https://doi.org/10.1145/3550469.3555378. 
Salter, S., Warren, R., Schlager, C., Spurr, A., Han, S., Bhasin, R., Cai, Y., Walkington, P., Bolarinwa, A., Wang, R., Danielson, N., Merel, J., Pnevmatikakis, E., & Marshall, J. (2024). emg2pose: A large and diverse benchmark for surface electromyographic hand pose estimation. arXiv. https://doi.org/10

## Repository Structure

```
.
├── model.py            # models (EMGPoseLSTM, SequentialEMGPoseLSTM, ...)
├── train.py            # Training loop with checkpointing and early stopping
├── load_data.py        # Data loading pipeline and get_dataloaders()
├── welford.py          # Welford online normalization utilities
├── data/
│   ├── session.py      # HDF5 session reader, windowed dataset, z-score scaling
│   ├── transforms.py   # EMG transforms (extraction, augmentation, downsampling)
│   ├── alignment.py    # Temporal alignment utilities
│   └── utils.py        # Split loading, IK failure masking, downsampling
├── notebooks/
│   ├── emg_pose_lstm_colab.ipynb          # Colab notebook for LSTM training
│   ├── lr_search_colab.ipynb              # Learning rate search
│   ├── CNN_LSTM.ipynb                     # CNN+LSTM model experiments
│   ├── CNN_LSTM_diagnostics.ipynb         # CNN+LSTM diagnostics and analysis
│   ├── TDS_LSTM.ipynb                     # TDS+LSTM model experiments
│   ├── checkpoint_CNN-LSTM_TDS-LSTM.ipynb # Checkpoint evaluation for CNN+LSTM and TDS+LSTM
│   ├── Model Visualizations.ipynb         # Model output visualizations
│   ├── Model_generalization.ipynb         # Cross-user generalization analysis
│   ├── MAE-joint_anlge_references.ipynb   # MAE and joint angle reference analysis
│   ├── Metadata Analysis.ipynb            # Dataset metadata exploration
│   └── Welford stat.ipynb                 # Welford normalization statistics
└── README.md
```

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
- **Z-score normalization**: Per-user EMG scaling using Welford online normalization (see `welford.py`)
- **Augmentation**: Random channel rotation during training
