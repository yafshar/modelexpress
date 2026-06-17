# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accelerator backend abstraction for device-specific operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


NIXL_ACCELERATOR_MEM_TYPE = "VRAM"


class AcceleratorBackend(Protocol):
    """Boundary for torch device control and accelerator capabilities."""

    @property
    def name(self) -> str:
        """Backend family name for logs and capability policy, for example ``cuda``."""
        ...

    @property
    def torch_device_type(self) -> str:
        """Torch device type used to construct tensors, which may differ from ``name``."""
        ...

    @property
    def nixl_mem_type(self) -> str:
        """NIXL memory segment for accelerator memory."""
        ...

    def set_device(self, device_id: int) -> None:
        """Make ``device_id`` current for this backend."""
        ...

    def current_device(self) -> int:
        """Return the current local device ordinal."""
        ...

    def synchronize(self, device_id: int | None = None) -> None:
        """Synchronize backend work on ``device_id`` or the current device."""
        ...

    def empty_cache(self) -> None:
        """Release backend allocator cache where supported."""
        ...

    def torch_device(self, device_id: int) -> torch.device:
        """Return a torch device object for ``device_id``."""
        ...

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` lives on this backend's accelerator memory."""
        ...

    def supports_pool_reg(self) -> bool:
        """Return whether allocation-level NIXL pool registration is supported."""
        ...

    def supports_vmm_arena(self) -> bool:
        """Return whether the CUDA VMM arena fast path is supported."""
        ...

    def supports_gds(self) -> bool:
        """Return whether GPUDirect Storage loading is supported."""
        ...


@dataclass(frozen=True)
class CudaAcceleratorBackend:
    """CUDA implementation of the accelerator backend boundary."""

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def torch_device_type(self) -> str:
        return "cuda"

    @property
    def nixl_mem_type(self) -> str:
        return NIXL_ACCELERATOR_MEM_TYPE

    def set_device(self, device_id: int) -> None:
        torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return int(torch.cuda.current_device())

    def synchronize(self, device_id: int | None = None) -> None:
        if device_id is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device_id)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def torch_device(self, device_id: int) -> torch.device:
        return torch.device(self.torch_device_type, device_id)

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return bool(tensor.is_cuda)

    def supports_pool_reg(self) -> bool:
        return True

    def supports_vmm_arena(self) -> bool:
        return True

    def supports_gds(self) -> bool:
        return True


@dataclass(frozen=True)
class XpuAcceleratorBackend:
    """XPU implementation of the accelerator backend boundary."""

    @property
    def name(self) -> str:
        return "xpu"

    @property
    def torch_device_type(self) -> str:
        return "xpu"

    @property
    def nixl_mem_type(self) -> str:
        return NIXL_ACCELERATOR_MEM_TYPE

    def _xpu(self):
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            raise RuntimeError("torch.xpu is not available")
        return xpu

    def set_device(self, device_id: int) -> None:
        self._xpu().set_device(device_id)

    def current_device(self) -> int:
        return int(self._xpu().current_device())

    def synchronize(self, device_id: int | None = None) -> None:
        xpu = self._xpu()
        if device_id is None:
            xpu.synchronize()
            return

        try:
            xpu.synchronize(device_id)
        except TypeError:
            current_device = int(xpu.current_device())
            xpu.set_device(device_id)
            try:
                xpu.synchronize()
            finally:
                xpu.set_device(current_device)

    def empty_cache(self) -> None:
        empty_cache = getattr(self._xpu(), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def torch_device(self, device_id: int) -> torch.device:
        return torch.device(self.torch_device_type, device_id)

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == self.torch_device_type

    def supports_pool_reg(self) -> bool:
        return False

    def supports_vmm_arena(self) -> bool:
        return False

    def supports_gds(self) -> bool:
        return False


def _is_torch_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _supported_device_types() -> str:
    supported = ["cuda"]
    if _is_torch_xpu_available():
        supported.append("xpu")
    return ", ".join(supported)


def accelerator_backend_for(device: torch.device | str) -> AcceleratorBackend:
    """Return the backend implementation for ``device``.

    CUDA is always selectable. XPU is selectable when torch exposes an
    available ``torch.xpu`` runtime.
    """
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        return CudaAcceleratorBackend()
    if torch_device.type == "xpu":
        if _is_torch_xpu_available():
            return XpuAcceleratorBackend()
        raise ValueError(
            "Unsupported accelerator backend for torch device "
            f"{torch_device!s}: torch.xpu is not available; "
            f"supported device types: {_supported_device_types()}"
        )
    raise ValueError(
        "Unsupported accelerator backend for torch device "
        f"{torch_device!s}; supported device types: {_supported_device_types()}"
    )
