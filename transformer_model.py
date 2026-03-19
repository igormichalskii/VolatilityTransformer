import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class VolatilityTransformer(nn.Module):
    # THE FIX: We pass seq_length in so the matrix knows its own size
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, dropout=0.2, seq_length=7):
        super(VolatilityTransformer, self).__init__()
        self.model_type = 'Transformer'

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # THE DYNAMIC SQUEEZE
        self.decoder = nn.Linear(d_model * seq_length, 1)
    
    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        # THE FIX: Smesh the 7-hour matrix into a single flat line
        # Shape goes from (Batch, 7, 64) -> (Batch, 448)
        flattened_output = output.flatten(start_dim=1)

        # Now the 448-neuron matrix perfectly fits the 448-neuron decoder
        prediction = self.decoder(flattened_output)
        return prediction