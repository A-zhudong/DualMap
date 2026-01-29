import asyncio
import math
import time
from dataclasses import dataclass
from typing import List
from collections import deque
from dualmap.entities.request import Request
from dualmap.logger import init_logger
from dualmap.cache_manager.kvcache_store.prefix_kvcache_store import PrefixKvCacheStore

logger = init_logger(__name__)

@dataclass
class Replica:
    def __init__(self, id, tokenizer, args):
        self.id = id
        self.tokenizer = tokenizer
        self.replica_slo_budget = args.replica_slo_budget
        self.current_budget=args.replica_slo_budget
        self.received_first_req = False
        self.last_prefill_completed_at=-1
        self.last_ttft=0
        self.pending_requests: List[Request] = []
        self.running_requests: List[Request] = []
        self.lock=asyncio.Lock()
        self.cache=PrefixKvCacheStore(args)
        self.block_size=args.block_size
        self.request_timestamps = deque()
        self.window_size = 180  # 180s

    def get_num_cached_tokens(self, prompt_token_ids: List[int]):
        num_cached_blocks = self.cache.get_num_cached_blocks(prompt_token_ids)
        if num_cached_blocks > 0:
            return num_cached_blocks * self.block_size
        else:
            return -1

    def get_num_recompute_token_ids(self, prompt_token_ids: List[int]):
        num_cached_tokens = self.get_num_cached_tokens(prompt_token_ids)
        if num_cached_tokens > 0 and num_cached_tokens < len(prompt_token_ids):
            return len(prompt_token_ids) - num_cached_tokens
        else:
            return len(prompt_token_ids)

    def save_token_ids(self, token_ids: List[int]): 
        num_saved_blocks = self.cache.save(token_ids)
        return num_saved_blocks * self.block_size

    def save_prefill_token_ids(self, request_id, token_ids: List[int]): 
        num_saved_blocks = self.cache.save_prefill(request_id, token_ids)
        return num_saved_blocks * self.block_size

    async def get_current_budget(self) -> int:
        async with self.lock: 
            return self.current_budget

    def get_num_running_req(self):
        return len(self.running_requests)

    def get_num_running_req_info(self):
        infos = []
        for req in self.running_requests:
            if req is not None:
                infos.append([req._id, len(req._input_ids), req._output_len, round(time.perf_counter()-req._arrived_at,4)])        
        return infos

    def get_running_req_blocks_cnt(self):
        blocks_cnt = 0
        for req in self.running_requests:
            if req is not None:
                blocks_cnt += math.ceil((len(req._input_ids) + req._output_len) / self.block_size)
        return blocks_cnt

    def get_num_pending_req(self):
        return len(self.pending_requests)

    def get_num_pending_req_info(self):
        infos = []
        for req in self.pending_requests:
            if req is not None:
                infos.append([req._id, len(req._input_ids), req._output_len, round(time.perf_counter()-req._arrived_at,4)])        
        return infos

    def get_num_actual_pending_tokens(self):       
        return max(self.replica_slo_budget - self.current_budget, 0)

    async def get_load_states(self):
        qps = 0
        async with self.lock:  
            qps = self.get_request_rate()
            if self.last_prefill_completed_at == -1:
                self.last_prefill_completed_at = time.perf_counter()
        return self.current_budget, self.last_prefill_completed_at, self.last_ttft, len(self.pending_requests), qps

    async def complete_request_prefill(self, replica_id, request: Request, ttft: float) -> bool:
        if request is None:
            return False
        # save kv cache into replca's cache
        num_saved_toekens = self.save_prefill_token_ids(request._id, request._input_ids)
        # logger.debug(f"complete_request_prefill:num_saved_toekens={num_saved_toekens}")

        async with self.lock: 
            try:
                self.received_first_req = True
                if request in self.pending_requests:
                    self.current_budget += request._actual_num_prefill_tokens
                    self.pending_requests.remove(request)
                    self.last_prefill_completed_at = time.perf_counter()
                    self.last_ttft = round(ttft,4)
                    logger.debug(f'complete_request_prefill: replica_id=[{replica_id}], request={request._id}, current_budget={request._actual_num_prefill_tokens},'
                                f'last_prefill_completed_at={self.last_prefill_completed_at},last_ttft={self.last_ttft}')

                    # pending_ids = [req._id for req in self.pending_requests]
                    # logger.debug(f"complete_request_prefill - Pending request IDs:replica_id=[{replica_id}], request={request._id},: {pending_ids}")
                    pending_tokens = [req._actual_num_prefill_tokens for req in self.pending_requests]
                    # logger.debug(f"complete_request_prefill - pending_tokens:replica_id=[{self.id}], request={request._id}, num_pending_req={len(pending_tokens)}: {pending_tokens}")
                    logger.debug(f"complete_request_prefill - replica_id=[{self.id}], request={request._id},self.replica_slo_budget-sum_pending_tokens==current_budget:{(self.replica_slo_budget-sum(pending_tokens))==self.current_budget},{self.replica_slo_budget}-{sum(pending_tokens)}=={self.current_budget}")
                    if request not in self.running_requests:
                        self.running_requests.append(request)
                    return True
            except ValueError: 
                return False
            except Exception as e:
                logger.debug(f'complete_request_prefill:Exception: {e}')
        return False

    def complete_request_decode(self, request: Request, output_text: str):
        if request is None:
            return
        if request in self.running_requests:
            self.running_requests.remove(request)
        output_token_ids = self.tokenizer.encode(output_text)
        token_ids = request._input_ids + output_token_ids
        num_saved_toekens = self.save_token_ids(token_ids)
        logger.debug(f"complete_request_decode:num_saved_toekens={num_saved_toekens}")

    async def add_request(self, request: Request) -> bool:
        if request is None:
            return False
        success = False
        async with self.lock:
            try:
                if request not in self.pending_requests:
                    request._actual_num_prefill_tokens = self.get_num_recompute_token_ids(request._input_ids)
                    self.current_budget -= request._actual_num_prefill_tokens
                    self.pending_requests.append(request)
                    now = time.perf_counter()
                    self.request_timestamps.append(now)
                    self._cleanup_old_requests(now)
                    # pending_ids = [req._id for req in self.pending_requests]
                    # logger.debug(f"add_request - Pending request IDs:replica_id=[{self.id}], request={request._id},{pending_ids}")
                    pending_tokens = [req._actual_num_prefill_tokens for req in self.pending_requests]
                    # logger.debug(f"add_request - pending_tokens:replica_id=[{self.id}], request={request._id}, num_pending_req={len(pending_tokens)}: {pending_tokens}")
                    logger.debug(f"add_request - replica_id=[{self.id}], request={request._id},self.replica_slo_budget-sum_pending_tokens==current_budget:{(self.replica_slo_budget-sum(pending_tokens))==self.current_budget},{self.replica_slo_budget}-{sum(pending_tokens)}=={self.current_budget}")
                    success = True
            except ValueError:
                success = False
            except Exception as e:
                success = False
                logger.debug(f'add_request:Exception: {e}')
        return success

    async def abort_request(self, request: Request) -> bool:
        res = False
        if request is None:
            return False
        async with self.lock:  # 必须加锁保证原子性
            try:
                if request in self.pending_requests:
                    self.current_budget += request._actual_num_prefill_tokens
                    self.pending_requests.remove(request)
                    pending_ids = [req._id for req in self.pending_requests]
                    logger.debug(f"abort_request - Pending request IDs:replica_id=[{self.id}], request={request._id}: {pending_ids}")
                    pending_tokens = [req._actual_num_prefill_tokens for req in self.pending_requests]
                    logger.debug(f"abort_request - pending_tokens:replica_id=[{self.id}], request={request._id}, num_pending_req={len(pending_tokens)}: {pending_tokens}")
                    logger.debug(f"abort_request - replica_id=[{self.id}], request={request._id},self.replica_slo_budget-sum_pending_tokens==current_budget:{(self.replica_slo_budget-sum(pending_tokens))==self.current_budget},{self.replica_slo_budget}-{sum(pending_tokens)}=={self.current_budget}")
                    res = True
                if request in self.running_requests:
                    self.running_requests.remove(request)
                    res = True
            except ValueError:
                return False
            except Exception as e:
                logger.debug(f'abort_request:Exception: {e}')
        return res


    def _cleanup_old_requests(self, current_time: float):
        if not self.request_timestamps or len(self.request_timestamps) == 0:
            return
        cutoff = current_time - self.window_size
        while True:
            try:
                oldest_timestamp = self.request_timestamps[0] if self.request_timestamps and len(self.request_timestamps) else None
                if oldest_timestamp is None or oldest_timestamp >= cutoff:
                    break
                self.request_timestamps.popleft()
            except IndexError:
                break

    def get_request_rate(self) -> float:
        current_time = time.perf_counter()
        if not self.request_timestamps or len(self.request_timestamps) == 0:
            return 0.0
        try:
            oldest_timestamp = self.request_timestamps[0]
        except IndexError:
            return 0.0

        window_duration = min(
            current_time - oldest_timestamp,
            self.window_size
        )
        if window_duration <= 0:
            return 0.0
        return round(len(self.request_timestamps) / window_duration, 4)