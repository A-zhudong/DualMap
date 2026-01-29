import asyncio
from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger

logger = init_logger(__name__)

class AsyncAtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            self._value += 1

    async def get(self) -> int:
        async with self._lock:
            return self._value
class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, num_replicas, shared_state: SharedState, args):
        super().__init__(num_replicas)
        self._request_counter = AsyncAtomicCounter(0)
        self.shared_state = shared_state

    async def schedule(self, request: Request) -> int:
        current = await self._request_counter.get()
        replica_id = current % self._num_replicas
        await self._request_counter.increment()
        return replica_id
    
    def finish_request(
        self, func_output=None, text: str = None, input_ids=None
    ):
        return