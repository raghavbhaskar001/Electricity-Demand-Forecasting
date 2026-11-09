# Deep Learning Model Architectures

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

All three neural architectures are implemented in PyTorch (`src/models.py`) and accept an input tensor of shape `(batch_size, seq_len=16, num_features=C)` where $C \in \{1, 10\}$, producing a scalar next-interval prediction of shape `(batch_size, 1)`.

---

## 1. Architecture Summary & Exact Parameter Counts

| Model Identifier | Class Name (`src/models.py`) | Core Inductive Bias | Parameters ($C=1$, `consumption_only`) | Parameters ($C=10$, weather configs) |
| :--- | :--- | :--- | ---: | ---: |
| **`1D_CNN`** | `CNN1DForecaster` | Local shift-invariant temporal convolution across 3-step (45-min) receptive fields | **52,225** | **53,953** |
| **`LSTM`** | `LSTMForecaster` | Sequential gated recurrence (`hidden_size=64`, `num_layers=2`) | **52,545** | **54,849** |
| **`CNN_LSTM`** | `CNNLSTMForecaster` | Local 1D convolutional feature extraction followed by 2-layer LSTM recurrence | **69,057** | **70,785** |

Notice that `1D_CNN` (`52,225` parameters) and `LSTM` (`52,545` parameters) are matched within $0.6\%$ parameter capacity so performance differences reflect architectural inductive bias rather than model size.

---

## 2. Layer-by-Layer Specifications

### 2.1 `1D_CNN` (`CNN1DForecaster`)
The input tensor `(B, 16, C)` is transposed to channel-first format `(B, C, 16)` before passing through two 1D convolutional blocks and a fully connected regression head:
1. **Conv Block 1:**
   - `Conv1d(in_channels=C, out_channels=64, kernel_size=3, padding=1)` $\to$ `(B, 64, 16)`
   - `BatchNorm1d(64)` $\to$ `GELU()` $\to$ `Dropout(p=0.2)`
2. **Conv Block 2:**
   - `Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)` $\to$ `(B, 64, 16)`
   - `BatchNorm1d(64)` $\to$ `GELU()` $\to$ `Dropout(p=0.2)`
3. **Flatten & Regression Head:**
   - `Flatten()` $\to$ `(B, 1024)` (since $64 \times 16 = 1,024$)
   - `Linear(1024, 32)` $\to$ `GELU()` $\to$ `Linear(32, 1)`

### 2.2 `LSTM` (`LSTMForecaster`)
Processes `(B, 16, C)` sequentially through a stacked two-layer Long Short-Term Memory network:
1. **Recurrent Encoder:**
   - `LSTM(input_size=C, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)` $\to$ sequence output `(B, 16, 64)`
2. **Terminal Hidden State Selection & Regression Head:**
   - Extracts terminal step representation `out[:, -1, :]` $\to$ `(B, 64)`
   - `Dropout(p=0.2)` $\to$ `Linear(64, 32)` $\to$ `GELU()` $\to$ `Linear(32, 1)`

### 2.3 `CNN_LSTM` (`CNNLSTMForecaster`)
Combines local convolutional filtering with recurrent temporal state tracking without temporal pooling so full 15-minute step resolution ($L = 16$) is preserved for the LSTM stage:
1. **Convolutional Feature Extractor:**
   - Transpose `(B, 16, C)` $\to$ `(B, C, 16)`
   - `Conv1d(C, 64, kernel_size=3, padding=1)` $\to$ `BatchNorm1d(64)` $\to$ `GELU()`
   - `Conv1d(64, 64, kernel_size=3, padding=1)` $\to$ `BatchNorm1d(64)` $\to$ `GELU()`
   - Transpose back to sequence-first `(B, 16, 64)`
2. **Recurrent Stage:**
   - `LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)`
   - Extracts terminal step `lstm_out[:, -1, :]` $\to$ `(B, 64)`
3. **Regression Head:**
   - `Dropout(p=0.2)` $\to$ `Linear(64, 32)` $\to$ `GELU()` $\to$ `Linear(32, 1)`
