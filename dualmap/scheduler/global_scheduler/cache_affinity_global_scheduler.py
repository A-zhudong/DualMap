import uhashring
from dualmap.entities.request import Request
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.preble_global_scheduler_utils import PrebleGlobalSchedulerUtils
from dualmap.scheduler.utils.shared import SharedState

from dualmap.logger import init_logger

logger = init_logger(__name__)

class CacheAffinityGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, num_replicas, window_duration: int, shared_state: SharedState, args):
        super().__init__(num_replicas)
    
    def hash_function1(self, task_id, num_nodes):
        nodes = [str(i) for i in range(num_nodes)]
        hash_ring1 = uhashring.HashRing(nodes=nodes)
        return int(hash_ring1.get_node(str(task_id)))

    async def schedule(self, request: Request) -> int:
        shortest_prefix = request._hash_session_id.split("/")[0]
        replica_id = self.hash_function1(shortest_prefix, self._num_replicas)
        logger.info(f"schedule: {request._hash_session_id} -> {replica_id}")
        return replica_id

    def finish_request(
        self, func_output: RequestFuncOutput=None, text: str = None, input_ids=None
    ):
        return