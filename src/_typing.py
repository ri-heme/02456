__all__ = ["ExperimentVersion", "List", "Optimizer", "PathLike", "Tuple"]

import os
from typing import List, Tuple, Union

from torch.optim import Optimizer

ExperimentVersion = Union[int, str, None]
PathLike = Union[str, bytes, os.PathLike]
