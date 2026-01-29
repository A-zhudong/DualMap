import asyncio
import time
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger

logger = init_logger(__name__)

class RequestRouterProxy:
    def __init__(self, shared_state: SharedState, global_scheduler_type, balance_type, args):
        self.shared_state = shared_state
        self.global_scheduler = None
        self._running = False
        self._global_scheduler_type = global_scheduler_type
        self._result_path = args.result_path
        self.replica_selection_task = None
        self.cleanup_selection_task = None

    async def _process_replica_selection(self):
        if self.global_scheduler is None:
            return
        while self._running:
            request = await self.shared_state.runtime_request_queue.get()
            if request is None:
                continue
            if request._id % 10 == 0:
                await self.shared_state.metric_store.sync_cache()
                if self._global_scheduler_type != "dualmap":
                    self.shared_state.record_replica_num_pending_request(self._result_path, request._id)
                    self.shared_state.record_replica_num_pending_tokens(self._result_path, request._id)
                    self.shared_state.record_replica_num_pending_input_tokens(self._result_path, request._id)   
            runtime_id = await self.global_scheduler.schedule(request)    
            await self.shared_state.schedule_request_to_replica(request, runtime_id)

    # for preble
    async def _process_cleanup_selection(self):
        while self._running:
            output, text, input_ids = await self.shared_state.finished_requests_queue.get()
            if output and output.success:
                self.global_scheduler.finish_request(func_output=output, text=text, input_ids=input_ids)

    async def start(self):
        self._running = True
        self.replica_selection_task = asyncio.create_task(self._process_replica_selection())
        self.cleanup_selection_task = asyncio.create_task(self._process_cleanup_selection()) #

    async def stop(self):
        self._running = False
        if self.replica_selection_task:
            self.replica_selection_task.cancel()
        if self.cleanup_selection_task:
            self.cleanup_selection_task.cancel()