import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of [max_len, d_model] representing the positional endocing
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Inject the Sine and Cosine waves
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class VolatilityTransformer(nn.Module):
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(VolatilityTransformer, self).__init__()
        self.model_type = 'Transformer'

        # 1. Project the raw 3D financial data (Returns, Vol, VIX) into a wider dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Inject the concept of Time
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. The Multi-Head Attention Matrix
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. Squeeze the wide mathematical output back into a single volatility prediction
        self.decoder = nn.Linear(d_model, 1)
    
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)

        # Project and encode
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        # Pass through the Transformer
        output = self.transformer_encoder(src)

        # We only car about the prediction for the final day in the sequence
        final_day_output = output[:, -1, :]

        # Decode to a single percentage
        prediction = self.decoder(final_day_output)
        return prediction