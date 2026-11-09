"""
Long Short-Term Memory (LSTM) Recurrent Network for 15-Minute Electricity Demand Forecasting
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        """
        Multi-layer LSTM Recurrent Neural Network.
        Models sequential temporal dependencies and long-range autoregressive relationships.
        """
        def __init__(self, in_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.in_features = in_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 32),
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
            out, (h_n, c_n) = self.lstm(x)
            # Use output of the last time step
            last_step = out[:, -1, :]
            prediction = self.regressor(last_step)
            return prediction

else:
    class LSTMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMForecaster. Run on Google Colab or install torch.")

