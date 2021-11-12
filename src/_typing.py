__all__ = ["List", "Optimizer", "PathLike", "Tuple"]

import os
from typing import List, Tuple, Union

from torch.optim import Optimizer

PathLike = Union[str, bytes, os.PathLike]
