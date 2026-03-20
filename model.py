"""
Standalone LSTM models for EMG-to-pose prediction.

EMGPoseLSTM        — processes full sequences at once (for training).
SequentialEMGPoseLSTM — steps through one timestep at a time (for streaming/inference).
CNNPoseLSTM       -- Conv1d frontend + LSTM, processes full sequences (for training)
CNNPoseLSTM2d     -- Conv2d frontend + LSTM, convolves across electrodes AND time
CNNOnly           -- Conv2d (electrode × time), no LSTM (for ablation / comparison)
"""

import torch
import torch.nn as nn


class EMGPoseLSTM(nn.Module):
    """LSTM that maps EMG signals to joint angle predictions.

    Processes the full sequence in one shot. Use this for training.

    Input:  (B, C_in=16, T)  — EMG channels over time
    Output: (B, C_out=20, T) — joint angle predictions over time
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 20,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            in_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=16, T) from dataloader
        x = x.permute(0, 2, 1)          # → (B, T, 16)
        out, _ = self.lstm(x)            # → (B, T, hidden)
        out = self.head(out)             # → (B, T, 20)
        return out.permute(0, 2, 1)      # → (B, 20, T) matching target shape


class SequentialEMGPoseLSTM(nn.Module):
    """LSTM that steps through one timestep at a time.

    Maintains internal hidden state across calls to step().
    Call reset_state() between sequences/trajectories.

    Use this for streaming inference or autoregressive decoding.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 20,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            in_channels,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels),
        )
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_state(self):
        """Reset hidden state. Call between sequences/trajectories."""
        self.hidden = None

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single timestep.

        Args:
            x: (B, C_in) — EMG channels for one timestep.

        Returns:
            (B, C_out) — joint angle predictions for this timestep.
        """
        if self.hidden is None:
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (
                torch.zeros(*size, device=device),
                torch.zeros(*size, device=device),
            )

        out, self.hidden = self.lstm(x[:, None], self.hidden)  # (B, 1, hidden)
        return self.head(out[:, 0])                             # (B, C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a full sequence by stepping through each timestep.

        Args:
            x: (B, C_in=16, T) — EMG channels over time.

        Returns:
            (B, C_out=20, T) — joint angle predictions over time.
        """
        x = x.permute(0, 2, 1)  # → (B, T, C_in)
        self.reset_state()
        outputs = []
        for t in range(x.size(1)):
            outputs.append(self.step(x[:, t]))  # each is (B, C_out)
        self.reset_state()
        out = torch.stack(outputs, dim=2)       # → (B, C_out, T)
        return out

# ──────────────────────────────────────────────
# CNN + LSTM
# ──────────────────────────────────────────────

class CNNPoseLSTM(nn.Module):
    """
    One Conv1d layer extracts spatial features from the 16 EMG channels,
    then an LSTM processes the sequence over time

    Input:  (B, 16, T)  -- raw EMG at 2kHz
    Output: (B, 20, T)  -- joint angle predictions
    """

    def __init__(
        self,
        input_channels=16,   # EMG channels
        out_features=20,     # joint angles
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        filters=64,          # number of CNN output features
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2  # keeps time dimension the same

        # CNN: extract spatial patterns across the 16 channels at each timestep
        # input (B, 16, T) -> output (B, filters, T)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
        )

        # LSTM: model temporal dependencies
        # input (B, T, filters) -> output (B, T, hidden_size)
        self.lstm = nn.LSTM(
            filters,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Linear head: project to joint angles
        # input (B, T, hidden_size) -> output (B, T, 20)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        # x: (B, 16, T) from dataloader

        # Spatial extraction
        x = self.cnn(x)             # (B, filters, T)

        # CNN outputs (B, Channels, Time) but LSTM wants (B, Time, Features)
        # so we swap the last two axes
        x = x.permute(0, 2, 1)      # (B, T, filters)

        # Temporal processing
        x, _ = self.lstm(x)          # (B, T, hidden_size)

        # Project to joint angles
        x = self.fc(x)              # (B, T, 20)

        return x.permute(0, 2, 1)   # (B, 20, T) to match target shape


# ──────────────────────────────────────────────
# CNN (Conv2d) + LSTM
# ──────────────────────────────────────────────

class CNNPoseLSTM2d(nn.Module):
    """
    Conv2d frontend that convolves across both electrode channels and time,
    followed by LSTM for temporal modeling

    Same idea as CNNOnly's Conv2d stack but instead of a 1x1 head,
    the pooled features feed into an LSTM decoder

    Input:  (B, 16, T)  -- raw EMG at 2kHz
    Output: (B, 20, T)  -- joint angle predictions
    """

    def __init__(
        self,
        input_channels=16,
        out_features=20,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        filters=64,
        kernel_size=3,
        num_conv_layers=3,
        channel_kernel=3,
    ):
        super().__init__()

        # Conv2d stack -- same as CNNOnly
        layers = []
        in_ch = 1  # raw EMG reshaped to (B, 1, 16, T)
        for i in range(num_conv_layers):
            out_ch = filters
            t_pad = kernel_size // 2
            c_pad = channel_kernel // 2
            layers.extend([
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(channel_kernel, kernel_size),
                          padding=(c_pad, t_pad)),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        # LSTM: model temporal dependencies
        # input (B, T, filters) -> output (B, T, hidden_size)
        self.lstm = nn.LSTM(
            filters,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Linear head: project to joint angles
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        # x: (B, 16, T) from dataloader
        x = x.unsqueeze(1)        # (B, 1, 16, T) -- single channel 2D input
        x = self.cnn(x)           # (B, filters, 16, T)
        x = x.mean(dim=2)        # (B, filters, T) -- pool across electrodes

        # CNN outputs (B, Channels, Time) but LSTM wants (B, Time, Features)
        x = x.permute(0, 2, 1)   # (B, T, filters)

        # Temporal processing
        x, _ = self.lstm(x)       # (B, T, hidden_size)

        # Project to joint angles
        x = self.fc(x)           # (B, T, 20)

        return x.permute(0, 2, 1)  # (B, 20, T) to match target shape


class SequentialCNNPoseLSTM(nn.Module):
    """
    Same architecture as CNNPoseLSTM but steps through the LSTM
    one timestep at a time -- for streaming inference

    The CNN still runs on the full sequence (it's a Conv1d, not recurrent)
    Only the LSTM part steps sequentially
    """

    def __init__(
        self,
        input_channels=16,
        out_features=20,
        hidden_size=256,
        num_layers=2,
        filters=64,
        kernel_size=3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        padding = kernel_size // 2

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
        )

        self.lstm = nn.LSTM(
            filters, hidden_size, num_layers, batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, out_features)
        self.hidden = None

    def reset_state(self):
        self.hidden = None

    def step(self, x_t):
        """One LSTM step for a single timestep
        x_t: (B, filters) -- CNN features for one timestep
        returns: (B, 20) -- joint angle predictions
        """
        if self.hidden is None:
            B = x_t.size(0)
            device = x_t.device
            size = (self.num_layers, B, self.hidden_size)
            self.hidden = (
                torch.zeros(*size, device=device),
                torch.zeros(*size, device=device),
            )
        out, self.hidden = self.lstm(x_t[:, None], self.hidden)  # (B, 1, hidden)
        return self.fc(out[:, 0])                                 # (B, 20)

    def forward(self, x):
        """
        x: (B, 16, T) -- raw EMG
        returns: (B, 20, T) -- joint angles
        """
        # CNN runs on full sequence
        features = self.cnn(x)              # (B, filters, T)
        features = features.permute(0, 2, 1)  # (B, T, filters)

        # LSTM steps one at a time
        self.reset_state()
        outputs = []
        for t in range(features.size(1)):
            outputs.append(self.step(features[:, t]))  # each (B, 20)
        self.reset_state()

        return torch.stack(outputs, dim=2)   # (B, 20, T)


# ──────────────────────────────────────────────
# CNN ONLY (no LSTM -- for ablation)
# ──────────────────────────────────────────────

class CNNOnly(nn.Module):
    """
    Conv2d model that convolves across both electrode channels and time
    The 16 EMG channels are treated as a spatial dimension (electrode
    positions on the forearm ring) so the kernel slides over both
    neighboring electrodes and neighboring timesteps

    Input:  (B, 16, T)  -- raw EMG at 2kHz
    Output: (B, 20, T)  -- joint angle predictions
    """

    def __init__(
        self,
        input_channels=16,
        out_features=20,
        filters=64,
        kernel_size=3,
        num_conv_layers=3,
        channel_kernel=3,
    ):
        super().__init__()

        layers = []
        in_ch = 1  # raw EMG reshaped to (B, 1, 16, T)
        for i in range(num_conv_layers):
            out_ch = filters
            t_pad = kernel_size // 2
            c_pad = channel_kernel // 2
            layers.extend([
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(channel_kernel, kernel_size),
                          padding=(c_pad, t_pad)),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        # Pool across electrodes then project to joint angles
        # After conv stack: (B, filters, 16, T)
        # Mean over electrode dim: (B, filters, T)
        # 1x1 conv to joint angles: (B, 20, T)
        self.head = nn.Conv1d(filters, out_features, kernel_size=1)

    def forward(self, x):
        # x: (B, 16, T)
        x = x.unsqueeze(1)    # (B, 1, 16, T) -- single channel 2D input
        x = self.cnn(x)       # (B, filters, 16, T)
        x = x.mean(dim=2)     # (B, filters, T) -- pool across electrodes
        x = self.head(x)      # (B, 20, T)
        return x

        
# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

class TransposedLayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        return self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)


# ──────────────────────────────────────────────
# TDS BUILDING BLOCKS
# ──────────────────────────────────────────────

class TDSConv2dBlock(nn.Module):
    """Temporal convolution block from Hannun et al. (2019)."""

    def __init__(self, channels: int, width: int, kernel_width: int):
        super().__init__()
        assert kernel_width % 2, "kernel_width must be odd"
        self.conv2d = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_width),
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)
        self.channels = channels
        self.width = width

    def forward(self, x):
        B, C, T = x.shape
        residual = x
        x = x.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)
        T_out = x.shape[-1]
        x = x + residual[..., -T_out:]   # skip connection
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSFullyConnectedBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        residual = x
        x = self.fc(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        x = x + residual
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


# ──────────────────────────────────────────────
# TDS FEATURIZER
# ──────────────────────────────────────────────

class TDSFeaturizer(nn.Module):
    """
    Compresses raw sEMG (B, 16, T) at 2kHz down to features (B, C, T') at 50Hz.

    Mirrors the vemg2pose featurizer from the paper:
      - 3 strided Conv1d blocks (strides 5, 2, 4 → 40x downsample)
      - 4 TDS blocks
      - Final upsample to 50Hz
    """

    def __init__(
        self,
        in_channels: int = 16,
        feature_channels: int = 256,
        tds_channels: int = 16,
        tds_kernel_widths: tuple = (9, 9, 5, 5),
    ):
        super().__init__()

        # Three strided convolutions: 2kHz → 25Hz (total stride = 5×2×4 = 40)
        self.conv_blocks = nn.Sequential(
            self._conv1d(in_channels,      feature_channels, kernel=11, stride=5),
            self._conv1d(feature_channels, feature_channels, kernel=5,  stride=2),
            self._conv1d(feature_channels, feature_channels, kernel=17, stride=4),
        )

        # TDS blocks
        width = feature_channels // tds_channels
        tds_blocks = []
        for kw in tds_kernel_widths:
            tds_blocks.append(TDSConv2dBlock(tds_channels, width, kw))
            tds_blocks.append(TDSFullyConnectedBlock(feature_channels))
        self.tds_blocks = nn.Sequential(*tds_blocks)

        self.out_channels = feature_channels

    @staticmethod
    def _conv1d(in_c, out_c, kernel, stride):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=0),
            TransposedLayerNorm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, 16, T) at 2kHz
        x = self.conv_blocks(x)          # → (B, feature_channels, T/40)
        x = self.tds_blocks(x)           # → (B, feature_channels, T/40)
        # Upsample from ~25Hz to 50Hz
        x = F.interpolate(x, scale_factor=2.0, mode='linear', align_corners=False)
        return x                         # → (B, feature_channels, T/20)


# ──────────────────────────────────────────────
# TDS + LSTM MODELS
# ──────────────────────────────────────────────

class TDS_LSTM(nn.Module):
    """
    TDS featurizer + LSTM decoder for EMG-to-pose prediction.
    Processes the full sequence in one shot. Use this for training.

    Input:  (B, 16, T)  — raw EMG at 2kHz
    Output: (B, 20, T)  — joint angle predictions (upsampled back to input length)
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 20,
        hidden_size: int = 256, # The paper uses two hidden layers of 512
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_channels: int = 256,
    ):
        super().__init__()

        self.featurizer = TDSFeaturizer(
            in_channels=in_channels,
            feature_channels=feature_channels,
        )

        self.lstm = nn.LSTM(
            feature_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_in = x.shape[-1]

        # TDS featurizer: (B, 16, T) → (B, feature_channels, T')
        features = self.featurizer(x)

        # LSTM: (B, T', feature_channels) → (B, T', hidden)
        out, _ = self.lstm(features.permute(0, 2, 1))

        # Project: (B, T', hidden) → (B, T', 20)
        out = self.head(out) * 0.01

        # Upsample back to original length: (B, T', 20) → (B, 20, T)
        out = F.interpolate(out.permute(0, 2, 1), size=T_in, mode='linear', align_corners=False)
        return out


class SequentialTDS_LSTM(nn.Module):
    """
    TDS featurizer + LSTM decoder, stepping one timestep at a time.
    Use this for streaming inference or autoregressive decoding.

    NOTE: TDS featurizer still processes the full sequence (it's a CNN).
          Only the LSTM decoder steps sequentially.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 20,
        hidden_size: int = 256,
        num_layers: int = 2,
        feature_channels: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.featurizer = TDSFeaturizer(
            in_channels=in_channels,
            feature_channels=feature_channels,
        )

        self.lstm = nn.LSTM(
            feature_channels,
            hidden_size,
            num_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels),
        )

        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_state(self):
        self.hidden = None

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        One LSTM step given a pre-computed feature vector.

        Args:
            x_t: (B, feature_channels) — features for one timestep.
        Returns:
            (B, out_channels) — predictions for this timestep.
        """
        if self.hidden is None:
            B, device = x_t.size(0), x_t.device
            size = (self.num_layers, B, self.hidden_size)
            self.hidden = (
                torch.zeros(*size, device=device),
                torch.zeros(*size, device=device),
            )
        out, self.hidden = self.lstm(x_t[:, None], self.hidden)  # (B, 1, hidden)
        return self.head(out[:, 0])                               # (B, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) — raw EMG at 2kHz.
        Returns:
            (B, 20, T) — joint angle predictions.
        """
        T_in = x.shape[-1]

        # TDS runs on full sequence (CNN, not sequential)
        features = self.featurizer(x)          # (B, feature_channels, T')
        features = features.permute(0, 2, 1)   # (B, T', feature_channels)

        # LSTM steps sequentially
        self.reset_state()
        outputs = []
        for t in range(features.size(1)):
            outputs.append(self.step(features[:, t]))   # each (B, out_channels)
        self.reset_state()

        out = torch.stack(outputs, dim=2)               # (B, out_channels, T')

        # Upsample back to input length
        return F.interpolate(out, size=T_in, mode='linear', align_corners=False)
