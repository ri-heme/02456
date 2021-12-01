__all__ = ["Block", "LCLayer", "LCStack"]

# based on SplitLinear class from Arnór
# SEE: https://github.com/arnor-sigurdsson/EIR/blob/99baff355e8479e67122e89a901c387acdddaefc/eir/models/layers.py#L235


from math import ceil

import torch
from torch import nn
from torch.nn.modules import lazy
import torch.nn.functional as F


class Block(nn.Module):
    """Block consisting of a linear transformation, SiLU activation function,
    batch normalization, and dropout layer.


    Parameters
    ----------
    transform : torch.nn.Module
        Transformation to be applied to the input, e.g., `torch.nn.Linear`
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    dropout_rate : int
        Probability to randomly dropout output units
    """

    def __init__(
        self,
        transform: nn.Module,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            transform(in_features, out_features), nn.SiLU(), nn.LazyBatchNorm1d()
        )
        if dropout_rate > 0.0:
            self.block.add_module("3", nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
        # 1) Transpose & flatten => (batch size, in features)
        if x.ndim == 3:
            x = x.transpose(1, 2).flatten(start_dim=1)
        if x.ndim != 2:
            raise ValueError("Input should be 2D (batch size, in features).")
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


class LCStack(nn.Module):
    """Stack of blocks of configurable depth.

    Parameters
    ----------
    depth : int
        Number of blocks in the stack
    in_features : int
        Number of input chunk features of each LC layer
    out_features : int
        Number of output chunk features of each LC layer
    dropout_rate : int
        Probability to randomly dropout units after each block
    """

    def __init__(self, depth, in_features, out_features, dropout_rate=0.0):
        super().__init__()
        blocks = [
            Block(LCLayer, in_features, out_features, dropout_rate)
            for _ in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
