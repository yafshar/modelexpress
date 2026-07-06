# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for NIXL registration-handle lifecycle.

Guards against the scratch-buffer registration leak: register_tensors /
register_arena must record their register_memory handle so shutdown can
deregister it, and the receive_weights_scratch path must deregister its
temporary scratch registration as soon as the transfer completes (or
fails) instead of accumulating stale MRs over freed GPU memory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from modelexpress.nixl_transfer import NixlTransferManager


def _make_manager() -> NixlTransferManager:
    mgr = NixlTransferManager(agent_name="test", device_id=0)
    mgr._agent = MagicMock()
    mgr._agent.get_agent_metadata.return_value = b"metadata"
    return mgr


class TestRegistrationTracking:
    def test_register_tensors_records_handle(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "handle-1"

        mgr.register_tensors({"w": torch.zeros(4, dtype=torch.float32)})

        assert mgr._tensor_registrations == ["handle-1"]

    def test_shutdown_deregisters_recorded_handles(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "handle-1"
        mgr.register_tensors({"w": torch.zeros(4, dtype=torch.float32)})

        agent = mgr._agent
        mgr.shutdown()

        agent.deregister_memory.assert_called_once_with("handle-1")
        assert mgr._tensor_registrations == []

    def test_shutdown_deregisters_in_reverse_order(self):
        mgr = _make_manager()
        mgr._agent.register_memory.side_effect = ["h1", "h2"]
        mgr.register_tensors({"a": torch.zeros(1)})
        mgr.register_tensors({"b": torch.zeros(1)})

        deregistered: list[str] = []
        mgr._agent.deregister_memory.side_effect = deregistered.append
        mgr.shutdown()

        assert deregistered == ["h2", "h1"]

    def test_shutdown_survives_deregister_error(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "handle-1"
        mgr.register_tensors({"w": torch.zeros(1)})
        mgr._agent.deregister_memory.side_effect = RuntimeError("boom")

        # Must not raise; agent is still torn down.
        mgr.shutdown()
        assert mgr._agent is None
        assert mgr._tensor_registrations == []


class TestTemporaryRegisteredTensors:
    def test_deregisters_on_exit(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "scratch-handle"

        with mgr.temporary_registered_tensors({"s": torch.zeros(4)}):
            mgr._agent.deregister_memory.assert_not_called()

        mgr._agent.deregister_memory.assert_called_once_with("scratch-handle")

    def test_scratch_handle_not_recorded_for_shutdown(self):
        # Scratch registration is deregistered by the context manager;
        # it must NOT also be tracked in _tensor_registrations (that would
        # double-deregister at shutdown).
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "scratch-handle"

        with mgr.temporary_registered_tensors({"s": torch.zeros(4)}):
            pass

        assert mgr._tensor_registrations == []

    def test_deregisters_on_exception(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "scratch-handle"

        try:
            with mgr.temporary_registered_tensors({"s": torch.zeros(4)}):
                raise ValueError("transfer failed")
        except ValueError:
            pass

        mgr._agent.deregister_memory.assert_called_once_with("scratch-handle")

    def test_restores_persistent_tensor_state(self):
        mgr = _make_manager()
        mgr._agent.register_memory.return_value = "persistent"
        persistent = {"model": torch.zeros(4)}
        mgr.register_tensors(persistent)

        saved_tensors = mgr._tensors
        saved_descriptors = mgr._tensor_descriptors
        saved_mem_type = mgr._local_mem_type

        mgr._agent.register_memory.return_value = "scratch"
        with mgr.temporary_registered_tensors({"scratch": torch.zeros(8)}):
            assert "scratch" in mgr._tensors

        assert mgr._tensors is saved_tensors
        assert mgr._tensor_descriptors is saved_descriptors
        assert mgr._local_mem_type == saved_mem_type
        # Only the persistent registration remains tracked for shutdown.
        assert mgr._tensor_registrations == ["persistent"]
