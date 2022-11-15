import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SeqPositionalEncoding(nn.Module):
    positional_encoding: Tensor

    def __init__(
        self,
        embedding_dim: int = 256,
        max_length: int = 512,
        batch_first: bool = False,
        dropout: Optional[float] = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.batch_first = batch_first
        if dropout is not None and 0.0 < dropout < 1.0:
            self.dropout: Optional[nn.Dropout] = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.register_buffer(
            "positional_encoding", torch.zeros(max_length, embedding_dim)
        )
        self.reset_buffer()

    def reset_buffer(self) -> None:
        position = torch.arange(
            0, self.max_length, dtype=torch.float32
        ).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.embedding_dim)
        )
        self.positional_encoding[:, 0::2] = torch.sin(position * divisors)
        self.positional_encoding[:, 1::2] = torch.cos(position * divisors)
        if self.batch_first:
            self.positional_encoding.unsqueeze_(0)
        else:
            self.positional_encoding.unsqueeze_(1)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: (batch_size, seq_len, embedding_dim) if batch_first
                else (seq_len, batch_size, embedding_dim)
        """
        if self.batch_first:
            positional_encoding = self.positional_encoding[:, : input.size(1)]
        else:
            positional_encoding = self.positional_encoding[: input.size(0)]
        output = input + positional_encoding
        if self.dropout is not None:
            output = self.dropout(output)
        return output

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"embedding_dim={self.embedding_dim}",
                f"dropout={self.dropout.p if self.dropout is not None else None}",
                f"max_length={self.max_length}",
                f"batch_first={self.batch_first}",
            ]
        )


class CoordPositionalEncoding(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        use_log_sampling: bool = True,
        sampling_rate: int = 10,
        max_resolution: int = 1024,
    ) -> None:
        super().__init__()
        self.use_log_sampling = use_log_sampling
        self.max_resolution = max_resolution

        if self.use_log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0, math.log2(max_resolution), sampling_rate
            )
        else:
            freq_bands = torch.linspace(1, max_resolution, sampling_rate)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, input: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError
