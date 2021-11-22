__all__ = ["LCLayer"]

# based on SplitLinear class from Arnór
# SEE: https://github.com/arnor-sigurdsson/EIR/blob/99baff355e8479e67122e89a901c387acdddaefc/eir/models/layers.py#L235


from math import ceil

import torch
from torch import nn
from torch.nn.modules import lazy
import torch.nn.functional as F


class LCLayer(lazy.LazyModuleMixin, nn.Linear):
    """Splits the incoming data into so-called chunks and applies a linear
    transformation to each chunk.

    Parameters
    ----------
    in_chunk_features : int
        Number of input features per chunk
    out_chunk_features : int
        Number of output features per chunk
    bias : bool, optional
        Whether an addititve bias can be learned, False by default

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable weight of shape (in_chunk_features, num_chunks,
        out_chunk_features)
    bias : torch.nn.Parameter
        Learnable bias of the shape (out_features)
    """

    num_chunks = 0
    padding = 0

    def __init__(
        self, in_chunk_features: int, out_chunk_features: int, bias: bool = False
    ):
        super().__init__(0, 0, False)
        self.in_chunk_features = in_chunk_features
        self.out_chunk_features = out_chunk_features
        self.weight = nn.UninitializedParameter()
        if bias:
            self.bias = nn.UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, x: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = x.shape[1:].numel()
                self.num_chunks = ceil(self.in_features / self.in_chunk_features)
                self.out_features = self.num_chunks * self.out_chunk_features
                self.padding = (
                    self.num_chunks * self.in_chunk_features - self.in_features
                )
                self.weight.materialize(
                    (self.out_chunk_features, self.num_chunks, self.in_chunk_features)
                )
                if self.bias is not None:
                    self.bias.materialize((self.out_features))
                self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                """Input should have 3 dimensions corresponding to (batch size, elements, features)."""
            )
        # 1) Transpose & flatten => (batch size, in features)
        x = x.transpose(1, 2).flatten(start_dim=1)
        # 2) Pad
        x = F.pad(x, (0, self.padding, 0, 0))
        # 3) Reshape => (batch size, chunks, in chunk features)
        x = x.reshape(x.shape[0], self.num_chunks, self.in_chunk_features)
        # 4) Multiply X * W and sum along axis 2 (in chunk features [k]) =>
        #    (batch size [i], chunks [j], out chunk features [x])
        out = torch.einsum("ijk, xjk -> ijx", x, self.weight)
        # 5) Flatten => (batch size, out features)
        out = out.flatten(start_dim=1)
        # 6) Add bias
        if self.bias is not None:
            out += self.bias
        return out
