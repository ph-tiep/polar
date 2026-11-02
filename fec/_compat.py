"""
_compat.py
======================================================================
Lightweight PyTorch Block & Validation Compatibility Layer for `fec`.
Provides `Block` (wrapping `torch.nn.Module`) and `check_binary`,
`int_mod_2` so `fec.polar` and `fec.crc` execute natively on PyTorch.
======================================================================
"""

from typing import Optional
import torch
import torch.nn as nn


def check_binary(x: torch.Tensor, name: str = "bits", message: str = "Input must be binary.") -> None:
    """Validate that tensor contains only binary values {0, 1}."""
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError(f"{name}: {message}")


def int_mod_2(x: torch.Tensor) -> torch.Tensor:
    """Modulo-2 operation on integer or float tensors."""
    return torch.remainder(x.to(torch.int32), 2)


class Block(nn.Module):
    """PyTorch base block compatible with Sionna PHY FEC blocks."""

    def __init__(
        self,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._precision = precision or "single"
        if self._precision in ("double", "float64"):
            self._dtype = torch.float64
        else:
            self._dtype = torch.float32
        self._device = str(device) if device is not None else "cpu"
        self._built = False

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    def build(self, *input_shapes) -> None:
        pass

    def forward(self, *args, **kwargs):
        if not self._built and len(args) > 0 and hasattr(args[0], "shape"):
            shapes = [tuple(a.shape) for a in args if hasattr(a, "shape")]
            if len(shapes) == 1:
                self.build(shapes[0])
            else:
                self.build(*shapes)
            self._built = True
        return self.call(*args, **kwargs)

