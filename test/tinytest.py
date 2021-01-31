import torch
from typing import Union, List, Optional, Dict, Tuple
from enum import Enum

@torch.jit.script
def fn(x: Union[int, float]) -> str:
    if x % 2:
        return "foo"
    else:
        return "bar"

print(fn.schema)
