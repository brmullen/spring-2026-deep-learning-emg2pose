"""
Standalone LSTM models for EMG-to-pose prediction.

EMGPoseLSTM        — processes full sequences at once (for training).
SequentialEMGPoseLSTM — steps through one timestep at a time (for streaming/inference).
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
