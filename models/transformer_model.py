import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.frontend_kernel_size = int(getattr(cfg, "frontend_kernel_size", 7))
        self.frontend_stride = int(getattr(cfg, "frontend_stride", 4))
        frontend_channels = int(getattr(cfg, "frontend_channels", 64))
        d_model = int(getattr(cfg, "d_model", 256))
        nhead = int(getattr(cfg, "nhead", 8))
        num_layers = int(getattr(cfg, "num_layers", 4))
        dim_feedforward = int(getattr(cfg, "dim_feedforward", 1024))
        dropout = float(getattr(cfg, "dropout", 0.1))

        padding = self.frontend_kernel_size // 2
        self.frontend_padding = padding
        self.frontend = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=frontend_channels,
                kernel_size=self.frontend_kernel_size,
                stride=self.frontend_stride,
                padding=padding,
            ),
            nn.BatchNorm1d(frontend_channels),
            nn.ReLU(),
        )
        self.input_projection = nn.Linear(frontend_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, int(num_classes))

    def forward(self, x):
        t_steps, batch = x.shape[0], x.shape[1]
        x = x.reshape(t_steps, batch, -1).permute(1, 2, 0)
        x = self.frontend(x)
        x = x.permute(2, 0, 1)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.to(torch.long)
        lengths = (
            (lengths + 2 * self.frontend_padding - (self.frontend_kernel_size - 1) - 1)
            // self.frontend_stride
        ) + 1
        return torch.clamp(lengths, min=1)


class PositionalEncoding(nn.Module):
    """
    Adds position information to embeddings.
    Input/output shape: (T, N, d_model)
    Standard implementation: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
