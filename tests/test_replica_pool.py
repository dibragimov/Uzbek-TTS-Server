import asyncio, unittest
from navoiy_server.replica_pool import NoCapacity, ReplicaPool

class Pool(unittest.IsolatedAsyncioTestCase):
    async def test_least_loaded_then_least_served(self):
        p = ReplicaPool(["a", "b", "c"], max_active=2, max_queue=4)
        leases = [await p.acquire(1) for _ in range(3)]
        self.assertEqual(sorted(l.slot.endpoint for l in leases), ["a", "b", "c"])
        await leases[1].release()
        nxt = await p.acquire(1)
        self.assertEqual(nxt.slot.endpoint, leases[1].slot.endpoint)  # the now-idle one

    async def test_release_is_idempotent(self):
        p = ReplicaPool(["a"], max_active=1, max_queue=0)
        l = await p.acquire(1)
        await l.release(); await l.release()
        self.assertEqual(p.slots[0].active, 0)

    async def test_waiter_woken_by_release_in_fifo_spirit(self):
        p = ReplicaPool(["a"], max_active=1, max_queue=2)
        first = await p.acquire(1)
        waiter = asyncio.create_task(p.acquire(2))
        await asyncio.sleep(0.05)
        self.assertEqual(p.waiting, 1)
        await first.release()
        got = await asyncio.wait_for(waiter, 1)
        self.assertEqual((got.slot.active, p.waiting), (1, 0))

    async def test_queue_full_and_timeout(self):
        p = ReplicaPool(["a"], max_active=1, max_queue=1)
        await p.acquire(1)
        w = asyncio.create_task(p.acquire(0.2))
        await asyncio.sleep(0.02)
        with self.assertRaises(NoCapacity) as full:
            await p.acquire(1)
        self.assertIn("full", full.exception.detail)
        with self.assertRaises(NoCapacity) as late:
            await w
        self.assertIn("Timed out", late.exception.detail)
        self.assertEqual(p.waiting, 0)

    async def test_cancelled_waiter_does_not_leak_queue_slot(self):
        p = ReplicaPool(["a"], max_active=1, max_queue=1)
        await p.acquire(1)
        w = asyncio.create_task(p.acquire(5))
        await asyncio.sleep(0.02)
        w.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await w
        self.assertEqual(p.waiting, 0)

    async def test_waiters_fail_fast_when_last_replica_dies(self):
        p = ReplicaPool(["a"], max_active=1, max_queue=2)
        await p.acquire(1)
        w = asyncio.create_task(p.acquire(5))
        await asyncio.sleep(0.02)
        await p.mark_unhealthy(0)
        with self.assertRaises(NoCapacity) as ctx:
            await asyncio.wait_for(w, 1)
        self.assertIn("No healthy", ctx.exception.detail)

if __name__ == "__main__":
    unittest.main(verbosity=2)
