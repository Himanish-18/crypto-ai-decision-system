
import torch
import torch.nn as nn
import numpy as np

# v39 Deep Orderflow Transformer (DOT)
# Encodes rolling tick sequences (Price, Side, Size) into a dense vector.
# Input: [Batch, SeqLen=120, Features=3]
# Output: [Batch, EmbedDim=64]

class OrderflowTransformer(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(OrderflowTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1) # Probability of Up-Tick
        )
        
    def forward(self, x):
        # x: [Batch, SeqLen, Features]
        bs, seq_len, _ = x.shape
        
        # 1. Embed + Positional Encoding
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # 2. Transformer
        x = self.transformer_encoder(x)
        
        # 3. Pooling (Last Token)
        last_hidden = x[:, -1, :]
        
        # 4. Prediction
        out = torch.sigmoid(self.head(last_hidden))
        return out, last_hidden

    def get_embedding(self, x):
        with torch.no_grad():
            _, hidden = self.forward(x)
        return hidden.numpy()

if __name__ == "__main__":
    # Test
    model = OrderflowTransformer()
    dummy_input = torch.randn(16, 120, 3) # Batch 16, 120 Ticks, (Price, Side, Size)
    pred, hidden = model(dummy_input)
    print(f"DOT Test: Output Shape {pred.shape}, Hidden {hidden.shape}")
