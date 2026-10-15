"""Admission control and routing across Triton replicas.

A request takes a lease on the least-loaded healthy replica that is below its
session limit, waiting in a bounded queue otherwise. Leases are released
exactly once, by the task that owns the request (see api.py), so capacity
cannot leak when clients disconnect.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


class NoCapacity(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class ReplicaSlot:
    index: int
    endpoint: str
    max_active: int
    active: int = 0
    healthy: bool = True
    served: int = 0


class Lease:
    def __init__(self, pool: "ReplicaPool", slot: ReplicaSlot):
        self._pool = pool
        self.slot = slot
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._pool._release(self.slot)


class ReplicaPool:
    def __init__(self, endpoints: list[str], max_active: int, max_queue: int):
        if not endpoints:
            raise ValueError("at least one replica endpoint is required")
        self.slots = [ReplicaSlot(i, ep, max_active) for i, ep in enumerate(endpoints)]
        self.max_queue = max_queue
        self.waiting = 0
        self.peak_active = 0
        self._cond = asyncio.Condition()

    def _pick(self) -> ReplicaSlot | None:
        ready = [s for s in self.slots if s.healthy and s.active < s.max_active]
        if not ready:
            return None
        # Least loaded; ties go to the replica that has served least, which
        # spreads warm-up and wear evenly instead of always hitting replica 0.
        return min(ready, key=lambda s: (s.active, s.served))

    def _any_healthy(self) -> bool:
        return any(s.healthy for s in self.slots)

    def _take(self, slot: ReplicaSlot) -> Lease:
        slot.active += 1
        slot.served += 1
        self.peak_active = max(self.peak_active, sum(s.active for s in self.slots))
        return Lease(self, slot)

    async def acquire(self, timeout_s: float) -> Lease:
        async with self._cond:
            if not self._any_healthy():
                raise NoCapacity(503, "No healthy TTS replica")
            slot = self._pick()
            if slot is not None:
                return self._take(slot)
            if self.waiting >= self.max_queue:
                raise NoCapacity(503, "TTS queue is full")
            self.waiting += 1
            try:
                slot = await asyncio.wait_for(
                    self._cond.wait_for(lambda: self._pick() or (not self._any_healthy())),
                    timeout=timeout_s)
            except asyncio.TimeoutError:
                raise NoCapacity(503, "Timed out waiting for a TTS slot") from None
            finally:
                self.waiting -= 1
            if slot is True or not isinstance(slot, ReplicaSlot):
                raise NoCapacity(503, "No healthy TTS replica")
            return self._take(slot)

    async def _release(self, slot: ReplicaSlot) -> None:
        async with self._cond:
            slot.active -= 1
            self._cond.notify_all()

    async def mark_unhealthy(self, index: int) -> None:
        async with self._cond:
            self.slots[index].healthy = False
            self._cond.notify_all()  # waiters re-check; they fail fast if none left

    def snapshot(self) -> dict:
        return {
            "waiting": self.waiting,
            "max_queue": self.max_queue,
            "peak_active": self.peak_active,
            "replicas": [{"index": s.index, "healthy": s.healthy, "active": s.active,
                          "max_active": s.max_active, "served": s.served} for s in self.slots],
        }
