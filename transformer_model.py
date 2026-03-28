import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super(Time2Vec, self).__init__()
        self.d_model = d_model

        # 1. The Linear Trend (Dimension 0)
        self.w0 = nn.Parameter(torch.Tensor(1, 1))
        self.b0 = nn.Parameter(torch.Tensor(1, 1))

        # 2. The Periodic Frequencies (Dimensions 1 through d_model-1)
        self.w = nn.Parameter(torch.Tensor(d_model - 1, 1))
        self.b = nn.Parameter(torch.Tensor(d_model - 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the clock with random noise so the Optimizer has something to mold
        nn.init.uniform_(self.w0, -1, 1)
        nn.init.uniform_(self.b0, -1, 1)
        nn.init.uniform_(self.w, -1, 1)
        nn.init.uniform_(self.b, -1, 1)

    def forward(self, tau):
        # tau is the actual time sequence: [0, 1, 2, 3, 4, 5, 6]
        # Linear component: w0 * t + b0
        linear = tau.matmul(self.w0) + self.b0

        # Periodic component: sin(w * t + b)
        periodic = torch.sin(tau.matmul(self.w.t()) + self.b.t())

        # Combine the linear trend with the cyclical sin waves
        return torch.cat([linear, periodic], dim=-1)
    
class VolatilityTransformer(nn.Module):
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, dropout=0.2, seq_length=7):
        super(VolatilityTransformer, self).__init__()
        self.model_type = "Transformer"

        # The Financial Data Upscaler
        self.input_projection = nn.Linear(input_size, d_model)

        # THE UPGRADE: The Learnable Clock
        self.time2vec = Time2Vec(d_model)

        # The Alien Brain
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # The Final Squeeze
        self.decoder = nn.Linear(d_model * seq_length, 1)

    def forward(self, src):
        batch_size, seq_len, _ = src.size()

        # 1. Project financial data to 64 dimensions
        projected_src = self.input_projection(src)

        # 2. Generate the raw time vector (Hours 0 through 6)
        tau = torch.arange(seq_len, dtype=torch.float32, device=src.device)
        tau = tau.view(1, seq_len, 1).expand(batch_size, -1, -1)

        # 3. Calculate the fluid time frequencies
        time_embedding = self.time2vec(tau)

        # 4. Inject the learned time directly into the financial data
        src = projected_src + time_embedding

        # 5. Process through Multi-Head Attention
        output = self.transformer_encoder(src)

        # 6. Flatten and Predict
        flattened_output = output.flatten(start_dim=1)
        prediction = self.decoder(flattened_output)
        return prediction