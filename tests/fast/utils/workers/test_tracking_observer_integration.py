"""Composition preview only: tracking cleanup and observer PR4550 remain independent owners."""

import asyncio

import pytest
from tests.fast.utils.workers.fake_ray import EVENT_KILL, FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager_serve import _launch, _make_spec

from miles.utils import observers
from miles.utils.workers import ray_worker_manager as rwm
from miles.utils.workers.types import WorkerCommBackend

pytestmark = pytest.mark.asyncio


def cleanup_events(cluster: FakeRayCluster) -> list[str]:
    return [event for event in cluster.events if event in {"finish_tracking", "report_process_end", EVENT_KILL}]


@pytest.mark.parametrize(
    ("transport", "expected"),
    [
        (WorkerCommBackend.RAY, ["finish_tracking", "report_process_end", EVENT_KILL]),
        (WorkerCommBackend.RPC, ["report_process_end", EVENT_KILL]),
    ],
)
async def test_each_transport_preserves_its_cleanup_order(fake_ray_cluster, transport, expected):
    manager = await _launch([_make_spec()], comm_backend=transport, report_process_ends=True)
    await manager.stop_cells(["trainer-00000"])
    assert cleanup_events(fake_ray_cluster) == expected


@pytest.mark.parametrize("failed_method", ["finish_tracking", "report_process_end"])
async def test_one_cleanup_failure_does_not_skip_later_cleanup(fake_ray_cluster, failed_method):
    manager = await _launch([_make_spec()], report_process_ends=True)
    fake_ray_cluster.method_errors[failed_method] = RuntimeError("cleanup failed")
    await manager.stop_cells(["trainer-00000"])
    assert cleanup_events(fake_ray_cluster) == ["finish_tracking", "report_process_end", EVENT_KILL]


@pytest.mark.parametrize(
    "hung_methods", [["finish_tracking"], ["report_process_end"], ["finish_tracking", "report_process_end"]]
)
async def test_cleanup_timeouts_have_independent_bounded_attempts(fake_ray_cluster, monkeypatch, hung_methods):
    manager = await _launch([_make_spec()], report_process_ends=True)
    for method in hung_methods:
        fake_ray_cluster.handles[0].hanging_methods[method] = 3600
    monkeypatch.setattr(rwm, "_SHUTDOWN_TIMEOUT", 0.01)
    monkeypatch.setattr(observers, "END_PROCESS_TIMEOUT_S", 0.02)
    real_wait_for = asyncio.wait_for
    budgets = []

    async def record_budget(awaitable, *, timeout):
        budgets.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(rwm.asyncio, "wait_for", record_budget)
    await real_wait_for(manager.stop_cells(["trainer-00000"]), timeout=2)
    assert budgets == [0.01, 0.02]
    assert cleanup_events(fake_ray_cluster) == ["finish_tracking", "report_process_end", EVENT_KILL]


async def test_default_cleanup_assigns_each_owner_its_own_timeout(fake_ray_cluster, monkeypatch):
    manager = await _launch([_make_spec()], report_process_ends=True)
    real_wait_for = asyncio.wait_for
    budgets = []

    async def record_budget(awaitable, *, timeout):
        budgets.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(rwm.asyncio, "wait_for", record_budget)
    await manager.stop_cells(["trainer-00000"])
    assert budgets == [30, 30]
    assert cleanup_events(fake_ray_cluster) == ["finish_tracking", "report_process_end", EVENT_KILL]
