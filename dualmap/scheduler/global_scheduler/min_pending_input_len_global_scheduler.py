from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger

logger = init_logger(__name__)

class MinPendingInputGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, num_replicas, shared_state: SharedState, args):
        super().__init__(num_replicas)
        self.shared_state = shared_state

    async def schedule(self, request: Request) -> int:
        replica_id, min_tokens = self.shared_state.get_min_pending_input_tokens_replica()
        return replica_id

    def finish_request(self, func_output=None, text: str = None, input_ids=None):
        return
