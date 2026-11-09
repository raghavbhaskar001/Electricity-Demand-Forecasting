"""
Hybrid CNN–LSTM Network for 15-Minute Electricity Demand Forecasting
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CNNLSTMForecaster(nn.Module):
        """
        Hybrid 1D CNN + LSTM Neural Network.
        Combines 1D CNN for local feature and transient pattern extraction with an LSTM
        for sequential dependency modeling across the compressed representation.
        """
        def __init__(self, in_features: int, cnn_channels: int = 64, lstm_hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.in_features = in_features
            
            # CNN front-end feature extractor
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=in_features, out_channels=cnn_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # LSTM sequential recurrent layer
            self.lstm = nn.LSTM(
                input_size=cnn_channels,
                hidden_size=lstm_hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            
            # Regressor head
            self.regressor = nn.Sequential(
                nn.Linear(lstm_hidden, 32),
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
            # Conv1d expects (batch_size, in_channels, seq_len)
            x_conv = self.cnn(x.permute(0, 2, 1))
            
            # Permute back to (batch_size, seq_len, cnn_channels) for LSTM
            x_lstm = x_conv.permute(0, 2, 1)
            out, (h_n, c_n) = self.lstm(x_lstm)
            
            last_step = out[:, -1, :]
            prediction = self.regressor(last_step)
            return prediction

else:
    class CNNLSTMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CNNLSTMForecaster. Run on Google Colab or install torch.")

