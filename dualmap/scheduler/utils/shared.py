import asyncio
import re
import time
import os
import random
from typing import Dict, Optional
from dualmap.entities.replica import Replica
from dualmap.entities.request import Request
from dualmap.client.open_ai import async_send_request
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.logger import init_logger

logger = init_logger(__name__)

class SharedState:
    def set_scheduler_callback(self, scheduler_callback):
        self._scheduler_callback = scheduler_callback

    def __init__(self, metric_store, tokenizer, args):
        self.metric_store = metric_store
        self.tokenizer = tokenizer
        self.runtime_events = {} #  Dict[str, Tuple[asyncio.Event, str]], runtime_events[request._id] = (event, runtime_id)
        self.runtime_request_queue = asyncio.Queue() # Request
        self.finished_requests_queue = asyncio.Queue()
        self.posting_request_tasks = {} # {(req_id, target_ip_port): task}
        self.posting_request_tasks_lock = asyncio.Lock() #{req_id, task}
        self.replicas_ip_port = args.replicas_ip_port.split(',')
        self.num_replicas = len(self.replicas_ip_port)
        self.replica_budgets: Dict[str, Replica] = {}  # {replica_id: Replica}
        self.replica_slo_budget = args.replica_slo_budget
        self.tpct = getattr(args, "tpct", args.prefill_tpot)
        self.tprt = getattr(args, "tprt", 0.0)
        self._result_path = args.result_path
        self._model_name = args.model_name
        self.last_request_time: Optional[float] = None
        for replica_id in range(self.num_replicas):
            self.replica_budgets[replica_id] = Replica(replica_id, self.tokenizer, args)  

    async def get_min_ttft_replica(self, request: Request):
        chose_replica_id = -1
        target_actual_prefill_len = -1
        replicas_pending_waiting_budget = {}
        replica_ttft_list = {}
        actual_num_prefill_tokens_list = {}
        hit_tokens_list = {}
        for rid in range(self.num_replicas):
            replica = self.replica_budgets[rid]
            actual_num_prefill_tokens = replica.get_num_recompute_token_ids(request._input_ids)
            hit_tokens = replica.get_num_hit_token_ids(request._input_ids)
            current_budget, last_prefill_completed_at, last_ttft, num_pending_requests, qps = await replica.get_load_states()
            replicas_pending_waiting_budget[rid] = current_budget - actual_num_prefill_tokens
            logger.debug(f"req_id={request._id},replicas_pending_waiting_budget[{rid}]={replicas_pending_waiting_budget[rid]}={current_budget}-{actual_num_prefill_tokens}")
            actual_num_prefill_tokens_list[rid] = actual_num_prefill_tokens
            hit_tokens_list[rid] = hit_tokens
            waiting_tokens = self.replica_slo_budget - current_budget
            waiting_hit_tokens = replica.get_num_pending_hit_tokens()
            # TTFT = (waiting recompute + current recompute) * TPCT + (waiting hit + current hit) * TPRT
            replica_ttft_list[rid] = round(
                (waiting_tokens + actual_num_prefill_tokens) * self.tpct + (waiting_hit_tokens + hit_tokens) * self.tprt,
                4
            )

        if replica_ttft_list:
            min_ttft = min(replica_ttft_list.values())
            candidates = [rid for rid, ttft in replica_ttft_list.items() if ttft == min_ttft]
            rnd = random.Random(42)
            chose_replica_id = rnd.choice(candidates)
            target_actual_prefill_len = actual_num_prefill_tokens_list[chose_replica_id]
        else:
            chose_replica_id = -1  # or some other default value
        pending_tokens = self.replica_slo_budget - (replicas_pending_waiting_budget[chose_replica_id] + target_actual_prefill_len)
        logger.debug(
            f"req_id={request._id},chose replica_id={chose_replica_id},waiting_tokens={pending_tokens},"
            f"actual_prefill_len={target_actual_prefill_len},hit_tokens={hit_tokens_list.get(chose_replica_id, 0)},"
            f"ttft={replica_ttft_list.get(chose_replica_id, -1)}"
        )
        return chose_replica_id, target_actual_prefill_len

    async def abort_posting_request_tasks(self, replica_id, request: Request):
        target_ip_port = self.replicas_ip_port[replica_id]
        key = (request._id, target_ip_port)
        async with self.posting_request_tasks_lock:
            if key in self.posting_request_tasks:
                task = self.posting_request_tasks[key]
                if not task.done():
                    task.cancel()
                    logger.info(f"Abort:Cancelled task for request {request._id} on {target_ip_port}")
                else:
                    task.exception()
                del self.posting_request_tasks[key]
                logger.debug(f"Abort:Removed task for request {request._id} on {target_ip_port}")

    async def del_posting_request_tasks(self, request: Request, target_ip_port: str):
        key = (request._id, target_ip_port)
        async with self.posting_request_tasks_lock:
            if key in self.posting_request_tasks:
                task = self.posting_request_tasks[key]
                if not task.done():
                    task.cancel()
                    logger.info(f"Cancelled task for request {request._id} on {target_ip_port}")
                else:
                    task.exception()
                del self.posting_request_tasks[key]
                logger.debug(f"Removed task for request {request._id} on {target_ip_port}")

    async def on_request_complete(self, output: RequestFuncOutput, request: Request, target_ip_port: str):
        self.last_request_time = time.perf_counter()
        if request is None:
            logger.error(f"Completion error on {target_ip_port}, request is None")
            return
        request_id = getattr(request, "_id", None)
        try:
            await self.del_posting_request_tasks(request, target_ip_port)
            await self.finished_requests_queue.put((output, request._prompts, request._input_ids))
            if hasattr(self, '_scheduler_callback') and self._scheduler_callback is not None:
                await self._scheduler_callback(None)
        except Exception as e:
            logger.error(f"Completion error for {request_id} on {target_ip_port}: {e}")

    def get_max_cache_hit_replica(self, request: Request):
        replica_id = -1
        prefix_cache_hit_len_list = {} #{replica_id:actual_num_prefill_tokens}
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            actual_num_prefill_tokens = replica.get_num_recompute_token_ids(request._input_ids)
            prefix_cache_hit_len_list[replica_id] = max(0, len(request._input_ids) - actual_num_prefill_tokens)
        if prefix_cache_hit_len_list:
            replica_id = max(prefix_cache_hit_len_list.items(), key=lambda x: x[1])[0]
        else:
            replica_id = -1  # or some other default value
        return replica_id

    async def add_posting_request_tasks(self, replica_id, request: Request):
        self.last_request_time = time.perf_counter()
        logger.info(f"Adding task for request {request._id} to replica {replica_id},session_id={request._native_session_id}")
        target_ip_port = self.replicas_ip_port[replica_id]
        task_key = (request._id, target_ip_port)
        # logger.info(f"Creating task for {task_key}")
        max_cache_hit_replcia = self.get_max_cache_hit_replica(request)
        if max_cache_hit_replcia == replica_id:
            request._rounting_cache_hit_max = 1
        else:
            request._rounting_cache_hit_max = 0
        async with self.posting_request_tasks_lock:
            try:
                output_task = asyncio.create_task(
                    async_send_request(
                        metric_store=self.metric_store,
                        result_path=self._result_path,
                        model_name=self._model_name,
                        replica_id=replica_id,
                        native_session_id=request._native_session_id,
                        target_ip_port=target_ip_port,
                        request=request,
                        replica=self.replica_budgets[replica_id]
                    )
                )

                def handle_task_completion(task):
                    key = (request._id, target_ip_port)
                    try:
                        result = task.result()
                        asyncio.create_task(self.on_request_complete(result, request, target_ip_port))
                    except asyncio.CancelledError:
                        logger.info(f"Task {key} cancelled before completion")
                    except Exception as e:
                        logger.error(f"Task {key} failed: {str(e)}")
                    finally:

                        asyncio.create_task(self.del_posting_request_tasks(request, target_ip_port))
                
                output_task.add_done_callback(handle_task_completion)
                self.posting_request_tasks[task_key] = output_task
                # logger.debug(f"Registered task {task_key}")
                
            except Exception as e:
                logger.error(f"Task creation failed for {request._id}: {str(e)}")
                await self.del_posting_request_tasks(request, target_ip_port)
                raise

    async def schedule_request_to_replica(self, request, runtime_id): 
        event = None
        try:
            event, _ = self.runtime_events[request._id]
            self.runtime_events[request._id] = (event, runtime_id)
        except Exception as e:
            logger.error(f"Selection error: {e}")
            self.runtime_events[request._id] = (event, None)
        finally:
            if event is not None:
                event.set()
                self.runtime_request_queue.task_done()

    def record_replica_num_pending_request(self, base_path, cur_request_id):
        num_request_pending_list = []
        num_request_running_list = []
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            num_request_pending = len(replica.pending_requests)
            num_request_pending_list.append(num_request_pending)
            num_request_running_list.append(replica.get_num_running_req())

        os.makedirs(base_path, exist_ok=True)
        try:
            with open (f'{base_path}/number_pending_requests.log', "a+") as file:
                for replica_id in range(len(num_request_pending_list)):
                    file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_pending_requests,{num_request_pending_list[replica_id]},number_running_requests,{num_request_running_list[replica_id]}\n')
            file.close()
        except Exception as e:
            print(f'error:MetricsConfig:save number_pending_requests failed! {e}')    


    def record_replica_num_pending_tokens(self, base_path, cur_request_id):
        number_pending_tokens_list = []
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            number_pending_tokens = self.replica_slo_budget - replica.current_budget
            number_pending_tokens_list.append(number_pending_tokens)

        os.makedirs(base_path, exist_ok=True)
        try:
            with open (f'{base_path}/number_pending_tokens.log', "a+") as file:
                for replica_id in range(len(number_pending_tokens_list)):
                    file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_pending_tokens,{number_pending_tokens_list[replica_id]}\n')
            file.close()
        except Exception as e:
            print(f'error:MetricsConfig:save number_pending_tokens failed! {e}')

    def record_replica_num_pending_input_tokens(self, base_path, cur_request_id):
            number_input_tokens_list = []
            for replica_id in range(self.num_replicas):
                replica = self.replica_budgets[replica_id]
                total_input_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
                number_input_tokens_list.append(total_input_tokens)

            os.makedirs(base_path, exist_ok=True)
            try:
                with open(f'{base_path}/number_pending_input_tokens.log', "a+") as file:
                    for replica_id in range(len(number_input_tokens_list)):
                        file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_input_tokens,{number_input_tokens_list[replica_id]}\n')
            except Exception as e:
                print(f'error:MetricsConfig:save number_input_tokens failed! {e}')

    def get_min_pending_input_tokens_replica(self):
        min_tokens = None
        min_replica_id = None
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            total_input_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
            if min_tokens is None or total_input_tokens < min_tokens:
                min_tokens = total_input_tokens
                min_replica_id = replica_id
        return min_replica_id, min_tokens

    def get_pending_input_tokens_replica(self,replica_id):
        replica = self.replica_budgets[replica_id]
        total_input_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
        return total_input_tokens

    def get_num_actual_pending_tokens_replica(self,replica_id):
        replica = self.replica_budgets[replica_id]
        return replica.get_num_actual_pending_tokens()

    def dump_replica_queue_info(self, replica_id):
        replica = self.replica_budgets[replica_id]
        logger.debug(f"local_info:replica={replica_id},"
        f"num_pending_req={replica.get_num_pending_req()},{replica.get_num_pending_req_info()},"
        f"running_req_blocks_cnt={replica.get_running_req_blocks_cnt()},"
        f"num_running_req={replica.get_num_running_req()},{replica.get_num_running_req_info()},")
