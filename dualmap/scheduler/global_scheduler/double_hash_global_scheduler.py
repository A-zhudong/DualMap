from dualmap.entities.request import Request
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.double_hash_global_scheduler_utils import DoubleHashGlobalSchedulerUtils
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger
logger = init_logger(__name__)

class DoubleHashGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, num_replicas, window_duration: int, balance_type, shared_state: SharedState, args):
        # logger.info(f"DoubleHashGlobalScheduler init: num_replicas={num_replicas}, window_duration={window_duration}, balance_type={balance_type}")
        super().__init__(num_replicas)
        self._result_path = args.result_path
        self.shared_state = shared_state
        self.double_hash_util = DoubleHashGlobalSchedulerUtils(num_replicas, shared_state, args)
        self.shared_state.set_scheduler_callback(self.schedule)


    async def schedule(self, new_request: Request) -> int:
        if new_request is not None:
            logger.info(f"schedule:new_req={new_request._id}, {new_request._hash_session_id}")
            if new_request._id % 10 == 0:    
                self.double_hash_util.record_replica_num_pending_request(self._result_path, new_request._id)
                self.double_hash_util.record_replica_num_pending_tokens(self._result_path, new_request._id) 
                self.double_hash_util.record_replica_num_pending_input_tokens(self._result_path, new_request._id)               
            await self.double_hash_util.add_request_to_best_global_queue(new_request)
            self.double_hash_util.global_request_queue_dump_info()
        schedulable_waiting_req_list = await self.double_hash_util.get_schedulable_waiting_req_list()
        logger.debug(f"schedule:num_req={len(schedulable_waiting_req_list)}")
        for request in schedulable_waiting_req_list:
            if request is None:
                continue
            if request._primary_replica >= 0 and request._primary_replica < self._num_replicas:
                await self.shared_state.add_posting_request_tasks(request._primary_replica, request)
            else:
                await self.double_hash_util.add_request_to_best_global_queue(request)            
        return -1
    
    def finish_request(
        self, func_output: RequestFuncOutput=None, text: str = None, input_ids=None
    ):
        pass