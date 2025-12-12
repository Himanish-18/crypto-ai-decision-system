
import torch
import torch.nn as nn
import numpy as np

# v39 Macro Regime v3: Volatility LSTM
# Forecasts realized volatility clusters to predict high-risk regimes.

class VolLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super(VolLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1) # Predicted Volatility (Log Variance)
        )
        
    def forward(self, x):
        # x: [Batch, SeqLen=60, Features=5] (OHLCV Returns)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred_vol = self.head(last_hidden)
        return pred_vol

    def predict_regime(self, recent_data):
        """
        Returns: 'HIGH_VOL', 'LOW_VOL', or 'NORMAL'
        """
        with torch.no_grad():
            x = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)
            pred = self.forward(x).item()
            
        # Thresholds (Example)
        if pred > 0.02: return "HIGH_VOL"
        if pred < 0.005: return "LOW_VOL"
        return "NORMAL"

if __name__ == "__main__":
    model = VolLSTM()
    dummy = np.random.randn(60, 5)
    print(f"Regime: {model.predict_regime(dummy)}")
