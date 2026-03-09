import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNRNNModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        conv_channels = getattr(cfg, "conv_channels", [64, 128])
        if isinstance(conv_channels, int):
            conv_channels = [conv_channels]
        conv_channels = [int(v) for v in conv_channels]

        conv_kernel_size = int(getattr(cfg, "conv_kernel_size", 7))
        conv_stride = int(getattr(cfg, "conv_stride", 2))
        conv_dropout = float(getattr(cfg, "conv_dropout", 0.1))

        cell_type = str(getattr(cfg, "cell_type", "lstm")).lower()
        hidden_dim = int(getattr(cfg, "hidden_dim", 256))
        num_rnn_layers = int(getattr(cfg, "num_rnn_layers", 2))
        rnn_dropout = float(getattr(cfg, "rnn_dropout", 0.2))
        bidirectional = bool(getattr(cfg, "bidirectional", True))

        in_channels = 32
        conv_layers = []
        self._conv_layout = []
        for out_channels in conv_channels:
            padding = conv_kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=conv_kernel_size,
                        stride=conv_stride,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(conv_dropout),
                ]
            )
            self._conv_layout.append((conv_kernel_size, conv_stride, padding))
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            dropout=(rnn_dropout if num_rnn_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=False,
        )
        out_features = hidden_dim * (2 if bidirectional else 1)
        self.output = nn.Linear(out_features, int(num_classes))

    def forward(self, x):
        t_steps, batch = x.shape[0], x.shape[1]
        x = x.reshape(t_steps, batch, -1).permute(1, 2, 0)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.to(torch.long)
        for kernel, stride, padding in self._conv_layout:
            lengths = ((lengths + 2 * padding - (kernel - 1) - 1) // stride) + 1
        return torch.clamp(lengths, min=1)
