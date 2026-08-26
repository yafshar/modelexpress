# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest
import torch

from modelexpress_rl.train.adapter import TrainerStagingMode, WeightPayloadFormat
from modelexpress_rl.train.engines.fsdp.adapter import FSDPTrainerAdapter

from tests.conftest import MockAcceleratorBackend

ADAPTER = "modelexpress_rl.train.engines.fsdp.adapter"


class _Manager:
    agent_name = "trainer-r0"
    nixl_metadata = b"agent-metadata"
    listen_port = 19000

    def __init__(self):
        self.registered = []

    def register_tensors(self, tensors):
        self.registered.append(dict(tensors))
        return self.nixl_metadata


@pytest.fixture
def dist_ready(monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.dist.is_available", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.is_initialized", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.get_rank", lambda: 0)


@pytest.fixture
def cpu_backend(monkeypatch):
    """COPY_TO_DEVICE needs a backend; these tests run on CPU, which has none.

    The adapter derives the backend from the captured shards' device so the two
    cannot disagree, which is the bug class being fixed; that leaves the lookup as
    the seam to stub rather than a backend argument threaded through the
    constructor.
    """
    backend = MockAcceleratorBackend(torch_device_type="cpu")
    monkeypatch.setattr(f"{ADAPTER}.accelerator_backend_for", lambda _device: backend)
    return backend


def _adapter(manager=None):
    return FSDPTrainerAdapter(
        manager=manager or _Manager(), nixl_metadata_endpoint="host:1234"
    )


def _stage(adapter, state_dict, mode=TrainerStagingMode.IN_PLACE):
    return adapter.stage_shard(
        tensors=state_dict,
        staging_mode=mode,
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )


def test_requires_initialized_distributed_engine(monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.dist.is_available", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="distributed process group"):
        _adapter()


def test_source_slot_id_is_rank_stamped(dist_ready):
    assert _adapter().source_slot_id == "publisher:global-rank:0"


def test_bind_tensors_validates_state_dict_and_returns_rank_slot(dist_ready):
    adapter = _adapter()

    assert adapter.bind_tensors({"w": torch.ones(2, 4)}) == (
        "publisher:global-rank:0"
    )
    with pytest.raises(TypeError, match="state_dict"):
        adapter.bind_tensors([torch.ones(2, 4)])


def test_in_place_stage_registers_once(dist_ready):
    manager = _Manager()
    adapter = _adapter(manager)
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}

    staged = _stage(adapter, state_dict)

    assert staged.manifest.tensor_count == 1
    assert staged.manifest.total_bytes == 2 * 4 * 2  # bf16 elsize
    assert staged.manifest.transport == "NIXL"
    staged.publish_ready.wait()  # IN_PLACE performs no copy: no-op

    # Re-staging the same weights must not re-register (setup is one-time).
    _stage(adapter, state_dict)
    assert len(manager.registered) == 1


def test_in_place_rejects_a_moved_source(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    # Same name/shape/dtype but fresh storage: the registered address is stale.
    with pytest.raises(NotImplementedError, match="source storage moved"):
        _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})


def test_stage_rejects_a_changed_tensor_set(dist_ready):
    adapter = _adapter()
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}
    _stage(adapter, state_dict)

    state_dict["b"] = torch.ones(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tensor set changed"):
        _stage(adapter, state_dict)


def test_stage_rejects_a_changed_shard_geometry(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    # Same name, different local shape: geometry must stay fixed after initialize.
    with pytest.raises(ValueError, match="shard geometry changed"):
        _stage(adapter, {"w": torch.ones(4, 4, dtype=torch.bfloat16)})


def test_in_place_requires_wire_dtype_source(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="use COPY_TO_DEVICE to cast"):
        _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.float32)})


def test_in_place_requires_contiguous_source(dist_ready):
    adapter = _adapter()
    strided = torch.ones(4, 4, dtype=torch.bfloat16).t()
    with pytest.raises(NotImplementedError, match="contiguous"):
        _stage(adapter, {"w": strided})


def test_staging_mode_cannot_change_after_initialize(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    with pytest.raises(ValueError, match="initialized for"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            mode=TrainerStagingMode.COPY_TO_DEVICE,
        )


def test_unsupported_staging_mode_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="COPY_TO_HOST"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            mode=TrainerStagingMode.COPY_TO_HOST,
        )


def test_unsupported_payload_format_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="XOR_DELTA"):
        adapter.stage_shard(
            tensors={"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.XOR_DELTA,
        )


def test_non_dict_tensors_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(TypeError, match="state_dict"):
        _stage(adapter, [torch.ones(2, 4, dtype=torch.bfloat16)])


def test_copy_to_device_allocates_arenas_in_the_backend_alloc_scope(
    dist_ready, cpu_backend, monkeypatch
):
    """The arena allocator is a backend requirement, not a CUDA constant."""
    scoped = []

    @contextmanager
    def fake_scope(backend):
        scoped.append(backend)
        yield

    monkeypatch.setattr(f"{ADAPTER}.registered_buffer_alloc_scope", fake_scope)
    manager = _Manager()
    adapter = _adapter(manager)

    staged = _stage(
        adapter,
        {"w": torch.ones(2, 4, dtype=torch.float32)},
        mode=TrainerStagingMode.COPY_TO_DEVICE,
    )

    assert scoped == [cpu_backend]
    # The arena is the served buffer, cast to the wire dtype.
    (registered,) = manager.registered
    (arena,) = registered.values()
    assert arena.dtype is torch.bfloat16
    assert torch.equal(arena, torch.ones(2, 4, dtype=torch.bfloat16))
    assert staged.manifest.total_bytes == 2 * 4 * 2


def test_copy_to_device_fence_waits_on_the_backend(dist_ready, cpu_backend):
    """The arena copies are asynchronous, so the fence must really block.

    A guard that degrades the fence to a no-op off CUDA publishes buffers whose
    copies are still in flight, which corrupts silently instead of failing.
    """
    adapter = _adapter()

    staged = _stage(
        adapter,
        {"w": torch.ones(2, 4, dtype=torch.float32)},
        mode=TrainerStagingMode.COPY_TO_DEVICE,
    )

    assert cpu_backend.fence_calls == [None]  # CPU shards carry no ordinal
    assert cpu_backend.fence_waits == []
    staged.publish_ready.wait()
    assert cpu_backend.fence_waits == [None]


def test_copy_to_device_fence_covers_every_stage(dist_ready, cpu_backend):
    state_dict = {"w": torch.ones(2, 4, dtype=torch.float32)}
    adapter = _adapter()

    _stage(adapter, state_dict, mode=TrainerStagingMode.COPY_TO_DEVICE)
    _stage(adapter, state_dict, mode=TrainerStagingMode.COPY_TO_DEVICE)

    assert cpu_backend.fence_calls == [None, None]


def test_copy_to_device_rejects_shards_on_more_than_one_device(dist_ready, cpu_backend):
    adapter = _adapter()
    state_dict = {
        "w": torch.ones(2, 4, dtype=torch.float32),
        "b": torch.ones(4, dtype=torch.float32, device="meta"),
    }

    with pytest.raises(NotImplementedError, match="state_dict spans"):
        _stage(adapter, state_dict, mode=TrainerStagingMode.COPY_TO_DEVICE)


def test_copy_to_device_rejects_a_source_that_changed_device(dist_ready, cpu_backend):
    """The arenas and the fence are bound to the initialize-time device.

    Geometry is unchanged here, so the layout check passes; without a device check
    the copy would run cross-device into an arena whose fence no longer names the
    source's device.
    """
    adapter = _adapter()
    _stage(
        adapter,
        {"w": torch.ones(2, 4, dtype=torch.float32)},
        mode=TrainerStagingMode.COPY_TO_DEVICE,
    )

    with pytest.raises(ValueError, match="source moved from cpu to meta"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.float32, device="meta")},
            mode=TrainerStagingMode.COPY_TO_DEVICE,
        )


def test_in_place_rejects_a_source_that_changed_device(dist_ready):
    """IN_PLACE publishes a raw address, which is only meaningful per device.

    A migration usually trips ``_require_sources_pinned`` incidentally, because the
    pointer changes too - that is what this input would hit without the device
    guard, and the message then blames storage movement and suggests
    COPY_TO_DEVICE, which does not fix a device change. The guard names the actual
    cause, and covers the case the pointer comparison cannot distinguish: two
    devices' address spaces holding the same integer.
    """
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    with pytest.raises(ValueError, match="source moved from cpu to meta"):
        _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16, device="meta")})


def test_an_unchanged_device_keeps_staging(dist_ready, cpu_backend):
    """The guard must not fire on the normal repeated-step path."""
    adapter = _adapter()
    state_dict = {"w": torch.ones(2, 4, dtype=torch.float32)}

    for _ in range(3):
        _stage(adapter, state_dict, mode=TrainerStagingMode.COPY_TO_DEVICE)

    assert cpu_backend.fence_calls == [None, None, None]


def test_copy_to_device_rejects_a_state_dict_that_is_not_on_an_accelerator(dist_ready):
    """The arenas are RDMA targets, so a CPU state_dict has to fail by name.

    Deliberately without the ``cpu_backend`` fixture: the real backend lookup is
    what rejects CPU here, and the point is that COPY_TO_DEVICE says so instead
    of surfacing a bare "unsupported device type" from the lookup.
    """
    adapter = _adapter()

    with pytest.raises(NotImplementedError, match="CPU-offloaded state_dict"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.float32)},
            mode=TrainerStagingMode.COPY_TO_DEVICE,
        )


def test_in_place_needs_no_accelerator_backend(dist_ready, monkeypatch):
    """IN_PLACE allocates nothing and copies nothing, so it resolves no backend."""
    monkeypatch.setattr(
        f"{ADAPTER}.accelerator_backend_for",
        lambda _device: pytest.fail("IN_PLACE must not resolve a backend"),
    )
    adapter = _adapter()

    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})
