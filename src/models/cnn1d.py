"""
1D Convolutional Neural Network (1D CNN) for 15-Minute Electricity Demand Forecasting
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CNN1DForecaster(nn.Module):
        """
        Multi-layer 1D Temporal Convolutional Network.
        Extracts temporal n-gram features and local patterns across input lookback sequences.
        """
        def __init__(self, in_features: int, sequence_length: int = 16, conv_channels=(64, 128, 64), dropout: float = 0.2):
            super().__init__()
            self.in_features = in_features
            self.sequence_length = sequence_length
            
            # Convolutional feature extraction blocks
            layers = []
            current_in = in_features
            for out_ch in conv_channels:
                layers.extend([
                    nn.Conv1d(in_channels=current_in, out_channels=out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_in = out_ch
            self.conv_stack = nn.Sequential(*layers)
            
            # Adaptive pool to collapse sequence dimension
            self.pool = nn.AdaptiveAvgPool1d(1)
            
            # Regression head
            self.regressor = nn.Sequential(
                nn.Linear(conv_channels[-1], 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor of shape (batch_size, sequence_length, in_features)
            Returns:
                Tensor of shape (batch_size, 1) predicting demand at t+1.
            """
            # PyTorch Conv1d expects (batch_size, in_channels, seq_len)
            x_permuted = x.permute(0, 2, 1)
            features = self.conv_stack(x_permuted)
            pooled = self.pool(features).squeeze(-1)
            prediction = self.regressor(pooled)
            return prediction

else:
    class CNN1DForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CNN1DForecaster. Run on Google Colab or install torch.")

