import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        cell_type = str(getattr(cfg, "cell_type", "lstm")).lower()
        hidden_dim = int(getattr(cfg, "hidden_dim", 256))
        num_layers = int(getattr(cfg, "num_layers", 3))
        dropout = float(getattr(cfg, "dropout", 0.2))
        bidirectional = bool(getattr(cfg, "bidirectional", True))

        input_proj_dim = getattr(cfg, "input_proj_dim", None)
        input_proj_dim = (
            int(input_proj_dim)
            if input_proj_dim is not None
            else int(getattr(cfg, "input_dim", 64))
        )

        self.input_projection = nn.Linear(32, input_proj_dim)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            batch_first=False,
        )
        out_features = hidden_dim * (2 if bidirectional else 1)
        self.output = nn.Linear(out_features, int(num_classes))

    def forward(self, x):
        t_steps, batch = x.shape[0], x.shape[1]
        x = x.reshape(t_steps, batch, -1)
        x = self.input_projection(x)
        x, _ = self.rnn(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths
