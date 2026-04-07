import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalGRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        output_size = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )

    def forward(self, x, lengths):
        """
        x: [B, T, F]
        lengths: [B]
        returns: logits [B]
        """
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.gru(packed)

        if self.gru.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]

        logits = self.classifier(final_hidden).squeeze(1)
        return logits
