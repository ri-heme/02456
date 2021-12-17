__all__ = ["Block", "LCLayer", "LCStack", "make_2d"]

from functools import wraps
from math import ceil
from typing import List

import torch
from torch import nn
from torch.nn.modules import lazy
import torch.nn.functional as F


def make_2d(method):
    """Flattens 3D input, conserving batch size dimension."""

    @wraps(method)
    def decorator(self: nn.Module, x: torch.Tensor):
        if x.ndim == 3:
            x = x.transpose(dim0=-1, dim1=1).flatten(start_dim=1)
        return method(self, x)

    return decorator


def pad(method):
    """Pads 2D input."""

    @wraps(method)
    @make_2d
    def decorator(self: nn.Module, x: torch.Tensor):
        if x.shape[1:].numel() == self.observation_features:
            x = F.pad(x, (0, self.padding))
        return method(self, x)

    return decorator


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
        linear = None
        if transform.__name__ == "LazyLinear":
            linear = transform(out_features)
        else:
            linear = transform(in_features, out_features)
        self.block = nn.Sequential(linear, nn.SiLU(), nn.LazyBatchNorm1d())
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

    # based on SplitLinear class from Arnór
    # SEE: https://github.com/arnor-sigurdsson/EIR/blob/master/eir/models/layers.py

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
                num_chunks = ceil(self.in_features / self.in_chunk_features)
                self.out_features = num_chunks * self.out_chunk_features
                self.weight.materialize(
                    (self.out_chunk_features, num_chunks, self.in_chunk_features)
                )
                if self.bias is not None:
                    self.bias.materialize((self.out_features))
                self.reset_parameters()

    @make_2d
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_chunks = self.weight.shape[1]
        # padding = (num_chunks * self.in_chunk_features - in_features)
        padding = self.weight.shape[1:].numel() - x.shape[1:].numel()
        # 1) Transpose & flatten => (batch size, in features)
        # 2) Pad
        x = F.pad(x, (0, padding))
        # 3) Reshape => (batch size, chunks, in chunk features)
        x = x.reshape(x.shape[0], num_chunks, self.in_chunk_features)
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

    def __init__(
        self, depth: int, in_features: int, out_features: int, dropout_rate: float = 0.0
    ):
        super().__init__()
        blocks = [
            Block(LCLayer, in_features, out_features, dropout_rate)
            for _ in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LinearStack(nn.Module):
    """Stack of blocks of configurable depth.

    Parameters
    ----------
    num_units : list of int
        Number of output features of each linear layer
    dropout_rate : float
        Probability to randomly dropout units after each block
    """

    def __init__(self, num_units: List[int], dropout_rate: float = 0.0):
        super().__init__()
        blocks = [
            Block(nn.LazyLinear, None, out_features, dropout_rate)
            for out_features in num_units
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
