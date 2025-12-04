import torch
import torch.nn as nn
import numpy as np
import logging

# Setup Logging
logger = logging.getLogger("transformer_alpha")

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based Alpha Model for Time Series Forecasting.
    Encodes price/volume sequences to predict future returns.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, output_dim: int = 1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (Batch, Seq_Len, Features)
        Returns:
            output: (Batch, Output_Dim) - Probability of Up move
        """
        # 1. Embed & Positional Encode
        src = self.input_embedding(src) # (Batch, Seq, d_model)
        src = self.pos_encoder(src)
        
        # 2. Transformer Encode
        output = self.transformer_encoder(src) # (Batch, Seq, d_model)
        
        # 3. Pool (Take last time step)
        last_step = output[:, -1, :] # (Batch, d_model)
        
        # 4. Decode
        prediction = self.decoder(last_step)
        return self.sigmoid(prediction)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, d_model)
        x = x + self.pe[:x.size(1), :]
        return x

# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Config
    BATCH_SIZE = 32
    SEQ_LEN = 60
    INPUT_DIM = 10
    
    # Create Model
    model = TimeSeriesTransformer(input_dim=INPUT_DIM)
    model.eval()
    
    # Create Dummy Input
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    
    # Forward Pass
    with torch.no_grad():
        output = model(dummy_input)
        
    logger.info(f"Input Shape: {dummy_input.shape}")
    logger.info(f"Output Shape: {output.shape}")
    logger.info(f"Sample Output: {output[0].item():.4f}")
    
    if output.shape == (BATCH_SIZE, 1):
        logger.info("✅ Transformer Alpha Model Verified.")
    else:
        logger.error("❌ Output shape mismatch.")
