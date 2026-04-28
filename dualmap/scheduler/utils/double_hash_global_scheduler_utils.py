import uhashring
import random
import os
import math
from collections import deque
from dualmap.entities.request import Request
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger
import heapq
import time
logger = init_logger(__name__)

class GlobalRequestQueue:
    def __init__(self, num_replicas, use_priority_sort: bool = False):
        self.num_replicas = num_replicas
        self.queues = [[] for _ in range(num_replicas)]
        self.queues_global_actual_waiting_tokens_count = [0 for _ in range(num_replicas)]
        self.queues_global_input_waiting_tokens_count = [0 for _ in range(num_replicas)]
        self.queues_global_waiting_hit_tokens_count = [0 for _ in range(num_replicas)]
        self.use_priority_sort = use_priority_sort

    def _queue_priority_rank(self, request):
        if self.use_priority_sort:
            return getattr(request, "_priority_rank", 1)
        return 1

    def _get_request_actual_prefill_tokens(self, request, negtive_prefix_cache_hit_len=None):
        if negtive_prefix_cache_hit_len is not None:
            return request._num_prefill_tokens + negtive_prefix_cache_hit_len
        else:
            return request._num_prefill_tokens

    def get_max_waiting_delay(self, replica_id):
        queue = self.queues[replica_id]
        if not queue:
            return 0
        earliest_arrived_at = min(item[2] for item in queue)
        waiting_delay = round(time.perf_counter() - earliest_arrived_at, 2)
        return waiting_delay
    
    def get_global_actual_waiting_tokens_count(self,replica_id):
        return self.queues_global_actual_waiting_tokens_count[replica_id]

    def get_global_input_waiting_tokens_count(self,replica_id):
        return self.queues_global_input_waiting_tokens_count[replica_id]

    def get_global_waiting_hit_tokens_count(self, replica_id):
        return self.queues_global_waiting_hit_tokens_count[replica_id]

    def _recount_pending_tokens(self, replica_id):
        total = 0
        total_hit_tokens = 0
        for item in self.queues[replica_id]:
            priority_rank, negtive_prefix_cache_hit_len,_,_,req = item
            total += self._get_request_actual_prefill_tokens(req, negtive_prefix_cache_hit_len)
            total_hit_tokens += max(0, req._num_prefill_tokens - self._get_request_actual_prefill_tokens(req, negtive_prefix_cache_hit_len))
        self.queues_global_actual_waiting_tokens_count[replica_id] = total
        self.queues_global_waiting_hit_tokens_count[replica_id] = total_hit_tokens
        return total

    def get_queue_len(self, rid):
        queue_len = 0
        queue = self.queues[rid]
        if queue is not None:
            queue_len = len(queue)
        return queue_len

    def dump_info(self):
        for rid, queue in enumerate(self.queues):
            if len(queue) > 0:
                logger.debug(f"replica={rid},queue_len={len(queue)},pending_token={self.queues_global_actual_waiting_tokens_count[rid]}")
                sorted_queue = sorted(queue)
                for item in sorted_queue:
                    priority_rank, negtive_prefix_cache_hit_len, arrival_at, _, req = item
                    if req is not None:
                        logger.debug(f"priority:{priority_rank},{negtive_prefix_cache_hit_len},{arrival_at};req_id={req._id}")

    def dump_replica_queue_info(self, rid):
        queue = self.queues[rid]
        logger.debug(f"global_info:replica={rid},queue_len={len(queue)},pending_token={self.queues_global_actual_waiting_tokens_count[rid]}")
        sorted_queue = sorted(queue)
        for item in sorted_queue:
            priority_rank, negtive_prefix_cache_hit_len, arrival_at, _, req = item
            if req is not None:
                logger.debug(f"priority:{priority_rank},{negtive_prefix_cache_hit_len},{round(time.perf_counter()-arrival_at,4)};req_id={req._id}")

    def push(self, replica_id, request, prefix_cache_hit_len):
        priority_rank = self._queue_priority_rank(request)
        heapq.heappush(self.queues[replica_id], (priority_rank, -prefix_cache_hit_len, request._arrived_at, request._id, request))
        self.queues_global_actual_waiting_tokens_count[replica_id] += self._get_request_actual_prefill_tokens(request,-prefix_cache_hit_len)
        self.queues_global_waiting_hit_tokens_count[replica_id] += max(0, request._num_prefill_tokens - self._get_request_actual_prefill_tokens(request,-prefix_cache_hit_len))
        self.queues_global_input_waiting_tokens_count[replica_id] += len(request._input_ids)
        return

    def pop(self, replica_id):
        if self.queues[replica_id]:
            item = heapq.heappop(self.queues[replica_id])
            priority_rank, negtive_prefix_cache_hit_len, _, _, req = item
            self.queues_global_actual_waiting_tokens_count[replica_id] -= self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len)
            self.queues_global_waiting_hit_tokens_count[replica_id] -= max(0, req._num_prefill_tokens - self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len))
            self.queues_global_input_waiting_tokens_count[replica_id] -= len(req._input_ids)
            return req
        return None

    def peek(self, replica_id):
        if self.queues[replica_id]:
            return self.queues[replica_id][0][4]
        return None

    def is_empty(self, replica_id):
        return len(self.queues[replica_id]) == 0

    def get_all_requests(self, replica_id):
        return [item[4] for item in self.queues[replica_id]]

    def discard_expired(self, replica_id, discard_threshold):
        logger.debug(f"before discard_expired")
        now = time.perf_counter()
        new_queue = []
        discarded = []
        for item in self.queues[replica_id]:
            priority_rank, negtive_prefix_cache_hit_len, arrived_at, _, req = item
            if now - arrived_at > discard_threshold:
                discarded.append(req)
                logger.debug(f"discard: req_id={req._id},waiting_latency={round(now - arrived_at, 2)}")
                self.queues_global_actual_waiting_tokens_count[replica_id] -= self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len)
                self.queues_global_waiting_hit_tokens_count[replica_id] -= max(0, req._num_prefill_tokens - self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len))
                self.queues_global_input_waiting_tokens_count[replica_id] -= len(req._input_ids)
            else:
                new_queue.append(item)
        self.queues[replica_id] = new_queue
        logger.debug(f"discard_expired req_num:{len(discarded)}")
        return discarded

    def del_req(self, replica_id, request):
        if request is None:
            return False
        queue = self.queues[replica_id]
        for idx, item in enumerate(queue):
            if item[4] == request:
                priority_rank, negtive_prefix_cache_hit_len, _, _, req = item
                del queue[idx]
                heapq.heapify(queue)
                self.queues_global_actual_waiting_tokens_count[replica_id] -= self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len)
                self.queues_global_waiting_hit_tokens_count[replica_id] -= max(0, req._num_prefill_tokens - self._get_request_actual_prefill_tokens(req,negtive_prefix_cache_hit_len))
                self.queues_global_input_waiting_tokens_count[replica_id] -= len(req._input_ids)
                return True
        return False

    def pop_schedulable(self, replica_id, cur_replica_budget, max_pop_num=None):
        schedulable = []
        pop_count = 0
        while self.queues[replica_id]:
            if max_pop_num is not None and pop_count >= max_pop_num:
                break
            priority_rank, negtive_prefix_cache_hit_len, _, _, req = self.queues[replica_id][0]
            actual_num_prefill_tokens = self._get_request_actual_prefill_tokens(req, negtive_prefix_cache_hit_len)
            req = heapq.heappop(self.queues[replica_id])[4]
            schedulable.append(req)
            pop_count += 1
            cur_replica_budget -= actual_num_prefill_tokens
            self.queues_global_actual_waiting_tokens_count[replica_id] -= actual_num_prefill_tokens
            self.queues_global_waiting_hit_tokens_count[replica_id] -= max(0, req._num_prefill_tokens - actual_num_prefill_tokens)
            self.queues_global_input_waiting_tokens_count[replica_id] -= len(req._input_ids)
            logger.debug(f"pop_schedulable:replica_id={replica_id},cur_replica_budget={cur_replica_budget},req={req._id},"
                        f"input_len={len(req._input_ids)},_num_prefill_tokens={req._num_prefill_tokens},"
                        f"actual_num_prefill_tokens={actual_num_prefill_tokens},negtive_prefix_cache_hit_len={negtive_prefix_cache_hit_len}")
            if cur_replica_budget <= 0:
                break
        return schedulable

    def get_num_global_actual_waiting_tokens(self, replica_id, request, prefix_cache_hit_len):
        queue = self.queues[replica_id]
        new_item = (self._queue_priority_rank(request), -prefix_cache_hit_len, request._arrived_at, request._id, request)
        simulated_queue = sorted(queue + [new_item])
        total_tokens = 0
        for item in simulated_queue:
            priority_rank, negtive_prefix_cache_hit_len, _, _, req = item
            if req == request:
                break
            total_tokens += self._get_request_actual_prefill_tokens(req, negtive_prefix_cache_hit_len)
        return total_tokens

    def get_num_global_waiting_hit_tokens(self, replica_id, request, prefix_cache_hit_len):
        queue = self.queues[replica_id]
        new_item = (self._queue_priority_rank(request), -prefix_cache_hit_len, request._arrived_at, request._id, request)
        simulated_queue = sorted(queue + [new_item])
        total_hit_tokens = 0
        for item in simulated_queue:
            priority_rank, negtive_prefix_cache_hit_len, _, _, req = item
            if req == request:
                break
            actual_num_prefill_tokens = self._get_request_actual_prefill_tokens(req, negtive_prefix_cache_hit_len)
            total_hit_tokens += max(0, req._num_prefill_tokens - actual_num_prefill_tokens)
        return total_hit_tokens
    
class DoubleHashGlobalSchedulerUtils():
    def __init__(self, num_replicas, shared_state: SharedState, args):
        self._num_replicas = num_replicas
        self.shared_state = shared_state
        self._balance_type = args.balance_type
        self.dh_first_balance_ttft_thredhold = args.dh_first_balance_ttft_thredhold
        self.dh_rebalance_thredhold = args.dh_rebalance_thredhold
        self.dh_replica_pending_req_threshold = args.dh_replica_pending_req_threshold
        self.dh_rebalance_waiting_latency_thredhold = args.dh_rebalance_waiting_latency_thredhold
        self.prefill_tpot = args.prefill_tpot
        self.tpct = getattr(args, "tpct", self.prefill_tpot)
        self.tprt = getattr(args, "tprt", 0.0)
        self.local_hit_tprt = getattr(args, "local_hit_tprt", 0.0)
        self.remote_hit_tprt = getattr(args, "remote_hit_tprt", self.tprt)
        self.priority_enabled = getattr(args, "priority_enabled", True)
        self.priority_policy = getattr(args, "priority_policy", "quantile")
        self.priority_quantile = getattr(args, "priority_quantile", 0.99)
        self.priority_window_size = max(1, int(getattr(args, "priority_window_size", 256)))
        self.request_priority_affects_queue = getattr(args, "request_priority_affects_queue", False)
        self._priority_ttft_window = deque(maxlen=self.priority_window_size)
        self.replica_slo_budget = args.replica_slo_budget
        self.dh_recompute_punish_ratio = args.dh_recompute_punish_ratio
        self.rebalance_cnt = 0
        self.busy_prefill_interval = args.busy_prefill_interval
        self.global_request_queue = GlobalRequestQueue(
            num_replicas,
            use_priority_sort=self.request_priority_affects_queue,
        )

    def estimate_ttft(self, waiting_tokens, waiting_hit_tokens, hit_tokens, waiting_local_hit_tokens=0, local_hit_tokens=0):
        waiting_remote_hit_tokens = max(0, waiting_hit_tokens - waiting_local_hit_tokens)
        remote_hit_tokens = max(0, hit_tokens - local_hit_tokens)
        # Cost split used by scheduler scoring:
        # - miss/prefill tokens            -> tpct
        # - local reuse tokens             -> local_hit_tprt
        # - cross-instance reuse tokens    -> remote_hit_tprt
        return round(
            waiting_tokens * self.tpct
            + (waiting_local_hit_tokens + local_hit_tokens) * self.local_hit_tprt
            + (waiting_remote_hit_tokens + remote_hit_tokens) * self.remote_hit_tprt,
            4,
        )

    def estimate_recompute_latency(self, recompute_tokens, hit_tokens):
        return round(recompute_tokens * self.tpct + hit_tokens * self.tprt, 4)

    def _compute_priority_threshold(self):
        default_threshold = self.dh_first_balance_ttft_thredhold * self.tpct
        if self.priority_policy != "quantile" or len(self._priority_ttft_window) == 0:
            return default_threshold
        quantile = min(1.0, max(0.0, float(self.priority_quantile)))
        ordered = sorted(self._priority_ttft_window)
        idx = math.ceil(quantile * len(ordered)) - 1
        idx = max(0, min(idx, len(ordered) - 1))
        return ordered[idx]

    def _set_request_priority(self, request: Request, predicted_ttft: float):
        request._priority_enabled = int(bool(self.priority_enabled))
        request._priority_policy = self.priority_policy
        request._priority_quantile = self.priority_quantile
        request._priority_window_size = self.priority_window_size
        request._estimated_ttft = round(predicted_ttft, 4)
        high_priority_threshold = self._compute_priority_threshold()
        request._priority_threshold = round(high_priority_threshold, 4)
        if self.priority_enabled and predicted_ttft > high_priority_threshold:
            request._priority_level = "HIGH"
            request._priority_rank = 0
        else:
            request._priority_level = "LOW"
            request._priority_rank = 1
    
    def update_observed_ttft(self, observed_ttft: float):
        if observed_ttft is None:
            return
        try:
            ttft = float(observed_ttft)
        except (TypeError, ValueError):
            return
        if ttft < 0:
            return
        self._priority_ttft_window.append(round(ttft, 4))


    def _init_hash_rings(self, num_nodes):
        nodes = [str(i) for i in range(num_nodes)]
        self._hash_ring1 = uhashring.HashRing(nodes=nodes)
        self._hash_ring2 = uhashring.HashRing(nodes=nodes, hash_fn = 'ketama')

    def hash_function1(self, task_id, num_nodes):
        if not hasattr(self, '_hash_ring1') or len(self._hash_ring1.nodes) != num_nodes:
            self._init_hash_rings(num_nodes)
        return int(self._hash_ring1.get_node(str(task_id)))

    def hash_function2(self, task_id, num_nodes):
        if not hasattr(self, '_hash_ring2') or len(self._hash_ring2.nodes) != num_nodes:
            self._init_hash_rings(num_nodes)
        return int(self._hash_ring2.get_node(str(task_id)))

    def global_request_queue_dump_info(self):
        self.global_request_queue.dump_info()

    async def get_schedulable_waiting_req_list(self):
        schedulable_waiting_req_list = []
        rebalance_ttft_threshold = self.dh_rebalance_thredhold * self.tpct
        #rebalance
        for rid in range(self._num_replicas):
            max_waiting_delay = self.global_request_queue.get_max_waiting_delay(rid)
            global_waiting_tokens = self.global_request_queue.get_global_actual_waiting_tokens_count(rid)
            global_waiting_hit_tokens = self.global_request_queue.get_global_waiting_hit_tokens_count(rid)
            num_local_actual_pending_tokens = self.shared_state.get_num_actual_pending_tokens_replica(rid)
            replica = self.shared_state.replica_budgets[rid]
            num_local_pending_hit_tokens = replica.get_num_pending_hit_tokens()
            num_local_pending_local_hit_tokens = replica.get_num_pending_local_hit_tokens()
            pending_ttft = self.estimate_ttft(
                global_waiting_tokens + num_local_actual_pending_tokens,
                global_waiting_hit_tokens + num_local_pending_hit_tokens,
                0,
                waiting_local_hit_tokens=num_local_pending_local_hit_tokens,
            )
            if pending_ttft > rebalance_ttft_threshold \
                or max_waiting_delay >= self.dh_rebalance_waiting_latency_thredhold:
                logger.debug(f"rebalance:rid={rid},{max_waiting_delay} >= {self.dh_rebalance_waiting_latency_thredhold}")
                await self.rebalance_replica_global_waiting_reqs(
                    rid,
                    global_waiting_tokens + num_local_actual_pending_tokens - self.dh_rebalance_thredhold
                )

        for rid in range(self._num_replicas):
            replica = self.shared_state.replica_budgets[rid]
            cur_replica_budget, last_prefill_completed_at, last_ttft, num_pending_requests, qps = await replica.get_load_states()
            queue = self.global_request_queue.queues[rid]
            prefill_interval = round(time.perf_counter() - last_prefill_completed_at,4)
            if queue and len(queue) > 0:
                logger.debug(f"get_schedulable_waiting_req_list,rid={rid},len_queue={len(queue)}"
                f"cur_replica_budget={cur_replica_budget},prefill_interval={prefill_interval},num_pending_requests={num_pending_requests}")
            if queue and len(queue) > 0 and cur_replica_budget > 0:
                max_pop_num = self.dh_replica_pending_req_threshold - num_pending_requests
                if num_pending_requests == 0:
                    schedulable = self.global_request_queue.pop_schedulable(rid, cur_replica_budget, max_pop_num)
                    logger.debug(f"get_schedulable_waiting_req_list:rid={rid},{cur_replica_budget},pop_len={len(schedulable)} (idle)")
                    schedulable_waiting_req_list += schedulable
                elif (num_pending_requests <= self.dh_replica_pending_req_threshold and prefill_interval < self.busy_prefill_interval):
                    schedulable = self.global_request_queue.pop_schedulable(rid, cur_replica_budget, max_pop_num)
                    logger.debug(f"get_schedulable_waiting_req_list:rid={rid},{cur_replica_budget},pop_len={len(schedulable)} (active)")
                    schedulable_waiting_req_list += schedulable
        return schedulable_waiting_req_list
        
    async def add_request_to_best_global_queue(self, request: Request):
        if request is None:
            return
        primary_replica_id = -1
        second_replica_id = -1 
        #1 select replica
        chosen_replica_ids = []
        num_replicas = self._num_replicas
        shortest_prefix = request._hash_session_id.split("/")[0]
        replica_id1 = self.hash_function1(shortest_prefix, num_replicas)
        replica_id2 = self.hash_function2(shortest_prefix, num_replicas)
        if replica_id1 == replica_id2:
            replica_id2 = (replica_id1 + 1) % num_replicas

        chosen_replica_ids = []
        primary_replica_id = replica_id1
        second_replica_id = replica_id2

        if len(chosen_replica_ids) == 0:
            chosen_replica_ids.append(replica_id1)
            chosen_replica_ids.append(replica_id2)

        ttft_list = {} #{replica_id:ttft}
        req_qps_list = {} #{replica_id:qps}
        recompute_latency_list = {} 
        num_global_actual_waiting_tokens_list = {}
        num_global_waiting_hit_tokens_list = {}
        num_replica_actual_pending_tokens_list = {}
        num_replica_pending_hit_tokens_list = {}
        num_req_actual_prefill_tokens_list = {}
        num_req_hit_tokens_list = {}
        num_req_local_hit_tokens_list = {}
        prefix_cache_hit_len_list = {}

        cost_list = {} 
        pending_input_tokens_list = {}
        num_running_req_list = {}
        running_req_blocks_cnt_list = {}
        num_virtual_pending_tokens_list = {}
        max_waiting_delay_list = {}
        

        for replica_id in chosen_replica_ids:
            num_global_actual_waiting_tokens_list[replica_id] = self.global_request_queue.get_global_actual_waiting_tokens_count(replica_id)
            num_global_waiting_hit_tokens_list[replica_id] = self.global_request_queue.get_global_waiting_hit_tokens_count(replica_id)
            replica = self.shared_state.replica_budgets[replica_id]
            num_replica_actual_pending_tokens_list[replica_id] = self.shared_state.get_num_actual_pending_tokens_replica(replica_id)
            num_replica_pending_hit_tokens_list[replica_id] = replica.get_num_pending_hit_tokens()
            num_req_actual_prefill_tokens_list[replica_id] = replica.get_num_recompute_token_ids(request._input_ids)
            num_req_hit_tokens_list[replica_id] = replica.get_num_hit_token_ids(request._input_ids)
            num_req_local_hit_tokens_list[replica_id] = replica.get_num_local_hit_token_ids(request._input_ids)

            # for dualmap_least_loaded
            pending_input_tokens_list[replica_id] = self.global_request_queue.get_global_input_waiting_tokens_count(replica_id) + self.shared_state.get_pending_input_tokens_replica(replica_id)
            # for dualmap_cache_affinity, dualmap_no_rebalance
            prefix_cache_hit_len_list[replica_id] = max(0, len(request._input_ids) - num_req_actual_prefill_tokens_list[replica_id])

            # for dualmap_min_ttft, dualmap_no_rebalance
            num_virtual_pending_tokens_list[replica_id] = num_global_actual_waiting_tokens_list[replica_id] + \
                                                        num_replica_actual_pending_tokens_list[replica_id] + \
                                                        num_req_actual_prefill_tokens_list[replica_id]
            ttft_list[replica_id] = self.estimate_ttft(
                num_virtual_pending_tokens_list[replica_id],
                num_global_waiting_hit_tokens_list[replica_id] + num_replica_pending_hit_tokens_list[replica_id],
                num_req_hit_tokens_list[replica_id],
                waiting_local_hit_tokens=replica.get_num_pending_local_hit_tokens(),
                local_hit_tokens=num_req_local_hit_tokens_list[replica_id],
            )
            # for ["nb_cost1", "rb_cost1","rb_cost1_aggresive","rb_cost1_avg"]
            current_budget, last_prefill_completed_at, last_ttft, num_pending_requests, qps = await replica.get_load_states()
            req_qps_list[replica_id] = qps
            recompute_latency_list[replica_id] = self.estimate_recompute_latency(
                num_req_actual_prefill_tokens_list[replica_id],
                num_req_hit_tokens_list[replica_id],
            )
            cost_list[replica_id] = round(ttft_list[replica_id] + self.dh_recompute_punish_ratio * ttft_list[replica_id] * req_qps_list[replica_id] * recompute_latency_list[replica_id],4)

            # for rebalance 
            max_waiting_delay_list[replica_id] = self.global_request_queue.get_max_waiting_delay(replica_id)

            # replica running info 
            num_running_req_list[replica_id] = replica.get_num_running_req()
            running_req_blocks_cnt_list[replica_id] = replica.get_running_req_blocks_cnt()


        def select_replicas_based_on_metrics(
            metric_dict, 
            running_req_blocks_cnt_list, 
            chosen_replica_ids, 
            seed=42,
            primary_is_max=False 
        ):
            random.seed(seed) 
            
            replica_id1, replica_id2 = chosen_replica_ids
            
            if metric_dict[replica_id1] != metric_dict[replica_id2]:
                if primary_is_max:
                    primary_replica_id = max(metric_dict.items(), key=lambda x: x[1])[0]
                    second_replica_id = min(metric_dict.items(), key=lambda x: x[1])[0]
                else:
                    primary_replica_id = min(metric_dict.items(), key=lambda x: x[1])[0]
                    second_replica_id = max(metric_dict.items(), key=lambda x: x[1])[0]
            else: 
                if running_req_blocks_cnt_list[replica_id1] != running_req_blocks_cnt_list[replica_id2]:
                    primary_replica_id = min(running_req_blocks_cnt_list.items(), key=lambda x: x[1])[0]
                    second_replica_id = max(running_req_blocks_cnt_list.items(), key=lambda x: x[1])[0]
                else:
                    primary_replica_id, second_replica_id = random.sample(chosen_replica_ids, 2)
            
            return primary_replica_id, second_replica_id

        dh_type = self._balance_type
        first_balance_ttft_threshold = self.dh_first_balance_ttft_thredhold * self.tpct
        rebalance_ttft_threshold = self.dh_rebalance_thredhold * self.tpct
        if self._balance_type in ["dualmap_least_loaded"]:
            primary_replica_id, second_replica_id = select_replicas_based_on_metrics(
                pending_input_tokens_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=False
            )
        elif self._balance_type in ["dualmap_cache_affinity"]:
            primary_replica_id, second_replica_id = select_replicas_based_on_metrics(
                prefix_cache_hit_len_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=True
            )
        elif self._balance_type in ["dualmap_min_ttft"]:
            primary_replica_id, second_replica_id = select_replicas_based_on_metrics(
                ttft_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=False
            )

        elif self._balance_type in ["dualmap_no_rebalance","dualmap", "ttft_slo","ttft_avg"]:
            is_replica_overloaded = {}
            for replica_id in chosen_replica_ids:
                is_replica_overloaded[replica_id] = False
                if ttft_list[replica_id] > first_balance_ttft_threshold:
                    is_replica_overloaded[replica_id] = True

            cache_hit_high_rep_id, cache_hit_low_rep_id = select_replicas_based_on_metrics(
                prefix_cache_hit_len_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=True
            )
            if is_replica_overloaded[cache_hit_high_rep_id]: # cache_hit_high_rep_id is overloaded
                primary_replica_id, second_replica_id = select_replicas_based_on_metrics(
                    ttft_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=False
                )
                dh_type = f"{self._balance_type}:min_global_ttft"
                self.global_request_queue.dump_replica_queue_info(primary_replica_id)
                self.shared_state.dump_replica_queue_info(primary_replica_id)
                self.global_request_queue.dump_replica_queue_info(second_replica_id)
                self.shared_state.dump_replica_queue_info(second_replica_id)
                logger.debug("")
            else:
                primary_replica_id = cache_hit_high_rep_id
                second_replica_id = cache_hit_low_rep_id
                
        elif self._balance_type in ["nb_cost1", "rb_cost1","rb_cost1_aggresive","rb_cost1_avg"]:
            if cost_list:
                primary_replica_id, second_replica_id = select_replicas_based_on_metrics(
                    cost_list, running_req_blocks_cnt_list, chosen_replica_ids, primary_is_max=False
                )            
            else:
                raise RuntimeError(
                    f"cost_list is None."
                )
        assert(primary_replica_id != -1)
        assert(second_replica_id != -1)

        self.global_request_queue.dump_replica_queue_info(primary_replica_id)
        self.shared_state.dump_replica_queue_info(primary_replica_id)
        self.global_request_queue.dump_replica_queue_info(second_replica_id)
        self.shared_state.dump_replica_queue_info(second_replica_id)

        min_ttft = min(ttft_list.values())
        max_refix_cache_hit = max(prefix_cache_hit_len_list.values())
        is_primary_min_ttft = (ttft_list[primary_replica_id] == min_ttft)
        is_primary_cache_affinity = (prefix_cache_hit_len_list[primary_replica_id] == max_refix_cache_hit)
        is_primary_cache_affinity_and_least_loaded = is_primary_cache_affinity and is_primary_min_ttft
        request._is_dh_cache_affinity = int(is_primary_cache_affinity)
        request._is_dh_least_loaded = int(is_primary_min_ttft)
        request._is_dh_cache_affinity_least_loaded = int(is_primary_cache_affinity_and_least_loaded)

        self._set_request_priority(request, ttft_list[primary_replica_id])
        self.global_request_queue.push(primary_replica_id, request, prefix_cache_hit_len_list[primary_replica_id])
        logger.debug(f"add:req_id={request._id},session={request._native_session_id},actual_num_prefill_tokens={num_req_actual_prefill_tokens_list[primary_replica_id]} to global pool,primary_replica_id={primary_replica_id},second_replica_id={second_replica_id}")

        # for rebalnce
        is_replica_overloaded = {}
        num_pending_tokens_list = {}
        for replica_id in chosen_replica_ids:
            is_replica_overloaded[replica_id] = False
            pending_hit_tokens = num_global_waiting_hit_tokens_list[replica_id] + num_replica_pending_hit_tokens_list[replica_id]
            if replica_id == primary_replica_id:
                num_pending_tokens_list[replica_id] = num_virtual_pending_tokens_list[replica_id]
                pending_ttft = ttft_list[replica_id]
            else:
                num_pending_tokens_list[replica_id] = num_virtual_pending_tokens_list[replica_id] - num_req_actual_prefill_tokens_list[replica_id]
                pending_ttft = self.estimate_ttft(
                    num_pending_tokens_list[replica_id],
                    pending_hit_tokens,
                    0,
                )
            if pending_ttft > rebalance_ttft_threshold \
                or max_waiting_delay_list[replica_id] >= self.dh_rebalance_waiting_latency_thredhold:
                is_replica_overloaded[replica_id] = True

        if is_replica_overloaded[primary_replica_id] and is_replica_overloaded[second_replica_id]:
            for rep_id in [primary_replica_id,second_replica_id]:
                num_migrate_tokens = num_pending_tokens_list[rep_id] - self.dh_rebalance_thredhold
                logger.debug(f"rebalance_replica_global_waiting_reqs:source={rep_id},num_migrate_tokens={num_migrate_tokens},max_waiting_delay={max_waiting_delay_list[rep_id]}")
                await self.rebalance_replica_global_waiting_reqs(rep_id, num_migrate_tokens)

        request._primary_replica = primary_replica_id
        request._second_replica = second_replica_id
        return

    async def rebalance_replica_global_waiting_reqs(self, source_replica_id, num_target_migrate_prefill_tokens):
        if self._balance_type not in ["dualmap"]: # "ttft_slo_aggresive", "rb_cost1_aggresive", "ttft_avg", "rb_cost1_avg"
            return
        rebalance_ttft_threshold = self.dh_rebalance_thredhold * self.tpct
        global_num_request_waiting = self.global_request_queue.get_queue_len(source_replica_id)
        if global_num_request_waiting <= 1:
            return
        
        self.global_request_queue.dump_replica_queue_info(source_replica_id)
        self.shared_state.dump_replica_queue_info(source_replica_id)

        enable_migrate_to_neighbor_replica = False
        enable_ttft_avg= False
        if self._balance_type in ["dualmap"]: # "ttft_slo_aggresive", "rb_cost1_aggresive"
            enable_migrate_to_neighbor_replica = True
        if self._balance_type == ["ttft_avg", "rb_cost1_avg"]:
            enable_migrate_to_neighbor_replica = True
            enable_ttft_avg = True

        num_replicas = self._num_replicas
       
        num_global_actual_waiting_tokens_list = {}
        num_global_waiting_hit_tokens_list = {}
        num_replica_actual_pending_tokens_list = {}
        num_replica_pending_hit_tokens_list = {}
        num_req_actual_prefill_tokens_list = {}
        max_waiting_delay_list = {}
        req_qps_list = {} #{replica_id:qps}

        for replica_id in range(self._num_replicas):
            num_global_actual_waiting_tokens_list[replica_id] = self.global_request_queue.get_global_actual_waiting_tokens_count(replica_id)
            num_global_waiting_hit_tokens_list[replica_id] = self.global_request_queue.get_global_waiting_hit_tokens_count(replica_id)
            num_replica_actual_pending_tokens_list[replica_id] = self.shared_state.get_num_actual_pending_tokens_replica(replica_id)
            num_replica_pending_hit_tokens_list[replica_id] = self.shared_state.replica_budgets[replica_id].get_num_pending_hit_tokens()
            max_waiting_delay_list[replica_id] = self.global_request_queue.get_max_waiting_delay(replica_id)
            replica = self.shared_state.replica_budgets[replica_id]
            current_budget, last_prefill_completed_at, last_ttft, num_pending_requests, qps = await replica.get_load_states()
            req_qps_list[replica_id] = qps

        assert(req_qps_list)   

        source_replica = self.shared_state.replica_budgets[source_replica_id]
        
        cur_num_migrated_prefill_tokens = 0
        while True:
            requests_migrate_cost = {} #req_index:cost
            source_global_waiting_requests = [item[4] for item in self.global_request_queue.queues[source_replica_id]]
            for req_index in range(len(source_global_waiting_requests)):
                # source_cost 
                cur_request = source_global_waiting_requests[req_index]
                # source_prefill_time
                # source_actual_prefill_len
                source_actual_prefill_len = source_replica.get_num_recompute_token_ids(cur_request._input_ids)
                source_hit_tokens = source_replica.get_num_hit_token_ids(cur_request._input_ids)
                source_num_global_waiting_prefill_tokens = self.global_request_queue.get_num_global_actual_waiting_tokens(source_replica_id, cur_request, 
                                                                                                                         max(0,cur_request._num_prefill_tokens - source_actual_prefill_len))
                source_num_global_waiting_hit_tokens = self.global_request_queue.get_num_global_waiting_hit_tokens(
                    source_replica_id,
                    cur_request,
                    max(0, cur_request._num_prefill_tokens - source_actual_prefill_len),
                )

                source_waiting_tokens = num_replica_actual_pending_tokens_list[source_replica_id] + source_num_global_waiting_prefill_tokens + source_actual_prefill_len
                source_waiting_hit_tokens = num_replica_pending_hit_tokens_list[source_replica_id] + source_num_global_waiting_hit_tokens
                source_ttft = self.estimate_ttft(source_waiting_tokens, source_waiting_hit_tokens, source_hit_tokens)
                source_recompute_latency = self.estimate_recompute_latency(source_actual_prefill_len, source_hit_tokens)
                source_num_delay_requests = max(req_qps_list[source_replica_id] * source_ttft, len(source_global_waiting_requests) - (req_index + 1)) 
                source_cost = source_ttft + self.dh_recompute_punish_ratio * source_num_delay_requests * source_recompute_latency
                
                # target_cost
                target_replica_id = -1
                shortest_prefix = cur_request._hash_session_id.split("/")[0]
                replica_id1 = self.hash_function1(shortest_prefix, num_replicas)
                replica_id2 = self.hash_function2(shortest_prefix, num_replicas)
                if replica_id1 == replica_id2:
                    replica_id2 = (replica_id1 + 1) % num_replicas
                logger.debug(f"{cur_request._hash_session_id}:replica_id1={replica_id1},replica_id2={replica_id2}")
            
                target_replica_id = replica_id2 if source_replica_id == replica_id1 else replica_id1

                if target_replica_id < 0 or target_replica_id >= num_replicas:
                        logger.debug(f"{target_replica_id} < 0 or {target_replica_id} >= {num_replicas}")
                        continue
                #  target_schedule_delay
                # insert cur_request to target_replica tail
                target_replica = self.shared_state.replica_budgets[target_replica_id]
                target_actual_prefill_len = target_replica.get_num_recompute_token_ids(cur_request._input_ids)
                target_hit_tokens = target_replica.get_num_hit_token_ids(cur_request._input_ids)

                # target_num_waiting_token = num_global_actual_waiting_tokens_list[target_replica_id]
                target_virtual_pending_tokens = num_global_actual_waiting_tokens_list[target_replica_id] + \
                                                num_replica_actual_pending_tokens_list[target_replica_id] + \
                                                target_actual_prefill_len
                
                # check target_replica load
                target_waiting_hit_tokens = num_global_waiting_hit_tokens_list[target_replica_id] + num_replica_pending_hit_tokens_list[target_replica_id]
                target_overload_ttft = self.estimate_ttft(target_virtual_pending_tokens, target_waiting_hit_tokens, target_hit_tokens)
                if  source_replica_id == target_replica_id or target_overload_ttft > rebalance_ttft_threshold \
                    or max_waiting_delay_list[target_replica_id] >= self.dh_rebalance_waiting_latency_thredhold: 
                    logger.debug(f"target_replica_id={target_replica_id},target_overload_ttft={target_overload_ttft},or"
                            f"{max_waiting_delay_list[target_replica_id]}>{self.dh_rebalance_waiting_latency_thredhold}")
                    # target_replica is overloaded
                    if enable_migrate_to_neighbor_replica is False:
                        logger.debug(f"enable_migrate_to_neighbor_replica is False")
                        continue
                    else:#dh_ttft_slo_aggresive:to try to migrate to other replicas
                        tmp_target_replica_id = (source_replica_id + 1) % num_replicas
                        if tmp_target_replica_id == source_replica_id or tmp_target_replica_id == target_replica_id:
                            tmp_target_replica_id = (target_replica_id + 1) % num_replicas
                        if tmp_target_replica_id == source_replica_id or tmp_target_replica_id == target_replica_id:
                            logger.debug(f"tmp_target_replica_id == source_replica_id")
                            continue

                        tmp_target_replica = self.shared_state.replica_budgets[tmp_target_replica_id]
                        tmp_target_actual_prefill_len = tmp_target_replica.get_num_recompute_token_ids(cur_request._input_ids)
                        tmp_target_virtual_pending_tokens = num_global_actual_waiting_tokens_list[tmp_target_replica_id] + \
                                                        num_replica_actual_pending_tokens_list[tmp_target_replica_id] + \
                                                        tmp_target_actual_prefill_len
                        tmp_target_hit_tokens = tmp_target_replica.get_num_hit_token_ids(cur_request._input_ids)
                        tmp_target_waiting_hit_tokens = num_global_waiting_hit_tokens_list[tmp_target_replica_id] + num_replica_pending_hit_tokens_list[tmp_target_replica_id]
                        tmp_target_overload_ttft = self.estimate_ttft(tmp_target_virtual_pending_tokens, tmp_target_waiting_hit_tokens, tmp_target_hit_tokens)
                        if tmp_target_overload_ttft > rebalance_ttft_threshold \
                            or max_waiting_delay_list[tmp_target_replica_id] >= self.dh_rebalance_waiting_latency_thredhold: 
                            logger.debug(f"tmp_target_replica_id={tmp_target_replica_id},tmp_target_overload_ttft={tmp_target_overload_ttft},or"
                                        f"{max_waiting_delay_list[tmp_target_replica_id]} >= {self.dh_rebalance_waiting_latency_thredhold}")
                            continue
                        else:
                            target_replica_id = tmp_target_replica_id                      
                            
                # insert cur_request to target_replica tail
                target_replica = self.shared_state.replica_budgets[target_replica_id]
                target_actual_prefill_len = target_replica.get_num_recompute_token_ids(cur_request._input_ids)
                target_num_global_actual_waiting_tokens = self.global_request_queue.get_num_global_actual_waiting_tokens(target_replica_id, cur_request, 
                                                                                                                         max(0,cur_request._num_prefill_tokens - target_actual_prefill_len))
                target_num_global_waiting_hit_tokens = self.global_request_queue.get_num_global_waiting_hit_tokens(
                    target_replica_id,
                    cur_request,
                    max(0, cur_request._num_prefill_tokens - target_actual_prefill_len),
                )
                target_virtual_pending_tokens = target_num_global_actual_waiting_tokens + \
                                                num_replica_actual_pending_tokens_list[target_replica_id] + \
                                                target_actual_prefill_len

                target_ttft = self.estimate_ttft(
                    target_virtual_pending_tokens,
                    num_replica_pending_hit_tokens_list[target_replica_id] + target_num_global_waiting_hit_tokens,
                    target_hit_tokens,
                )
                target_recompute_latency = self.estimate_recompute_latency(target_actual_prefill_len, target_hit_tokens)
                target_num_delay_requests = round(req_qps_list[target_replica_id] * target_ttft, 4) 
                target_cost = round(target_ttft + self.dh_recompute_punish_ratio * target_num_delay_requests * target_recompute_latency, 4)

                # get cost 
                cost = 0
                if self._balance_type in ["rb_cost1","rb_cost1_aggresive","rb_cost1_avg"]: 
                    cost = target_cost - source_cost
                elif self._balance_type in ["dualmap"]: # "ttft_slo","ttft_slo_aggresive","ttft_avg"
                    cost = target_ttft - source_ttft
                if req_index not in requests_migrate_cost:
                    requests_migrate_cost[req_index] = (cost, source_actual_prefill_len, target_replica_id, target_replica, target_actual_prefill_len)
            
            if not requests_migrate_cost:
                break
            # low cost first
            sorted_requests_migrate_cost = sorted(requests_migrate_cost.items(), key=lambda x: x[1][0]) 
            for req_index, (cost, source_actual_prefill_len, target_replica_id, _, target_actual_prefill_len) in sorted_requests_migrate_cost:   
                logger.debug(f"req:{source_global_waiting_requests[req_index]._id},{source_global_waiting_requests[req_index]._native_session_id},cost={cost},source_replica_id={source_replica_id},target_replica_id={target_replica_id},"
                                f"source_actual_prefill_len={source_actual_prefill_len},target_actual_prefill_len={target_actual_prefill_len}")

            migrate_req_index, (cost, source_actual_prefill_len, target_replica_id, _, target_actual_prefill_len) = sorted_requests_migrate_cost[0]
            if migrate_req_index < 0 or migrate_req_index >= len(source_global_waiting_requests):
                break
            
            # chedk break condition
            elif cost >= 0: 
                logger.debug(f"cost={cost}>=0")
                break
            elif num_target_migrate_prefill_tokens > 0:
                if cur_num_migrated_prefill_tokens >= num_target_migrate_prefill_tokens and enable_ttft_avg is False:
                    logger.debug(f"cur_num_migrated_prefill_tokens >= num_target_migrate_prefill_tokens and enable_ttft_avg is False")
                    break

            # start migrate
            migrate_request:Request = source_global_waiting_requests[migrate_req_index]
            self.global_request_queue.dump_replica_queue_info(target_replica_id)
            self.shared_state.dump_replica_queue_info(target_replica_id)             
            self.global_request_queue.del_req(source_replica_id, migrate_request)
            target_prefix_cache_hit_len = migrate_request._num_prefill_tokens - target_actual_prefill_len
            target_num_global_actual_waiting_tokens = self.global_request_queue.get_num_global_actual_waiting_tokens(
                target_replica_id,
                migrate_request,
                target_prefix_cache_hit_len,
            )
            target_num_global_waiting_hit_tokens = self.global_request_queue.get_num_global_waiting_hit_tokens(
                target_replica_id,
                migrate_request,
                target_prefix_cache_hit_len,
            )
            target_hit_tokens = self.shared_state.replica_budgets[target_replica_id].get_num_hit_token_ids(migrate_request._input_ids)
            target_virtual_pending_tokens = (
                target_num_global_actual_waiting_tokens
                + num_replica_actual_pending_tokens_list[target_replica_id]
                + target_actual_prefill_len
            )
            migrated_predicted_ttft = self.estimate_ttft(
                target_virtual_pending_tokens,
                num_replica_pending_hit_tokens_list[target_replica_id] + target_num_global_waiting_hit_tokens,
                target_hit_tokens,
            )
            self._set_request_priority(migrate_request, migrated_predicted_ttft)
            self.global_request_queue.push(target_replica_id, migrate_request,target_prefix_cache_hit_len)
            num_global_actual_waiting_tokens_list[target_replica_id] += target_actual_prefill_len
            cur_num_migrated_prefill_tokens += source_actual_prefill_len
            self.rebalance_cnt += 1
            migrate_request._primary_replica = target_replica_id
            migrate_request._second_replica = source_replica_id
        return

    def record_replica_num_pending_request(self, base_path, cur_request_id):
        num_request_pending_list = []
        num_request_running_list = []
        for replica_id in range(self._num_replicas):
            replica = self.shared_state.replica_budgets[replica_id]
            replica_num_request_pending = len(replica.pending_requests)
            global_num_request_waiting = self.global_request_queue.get_queue_len(replica_id)
            num_request_pending_list.append(replica_num_request_pending + global_num_request_waiting)
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
        for replica_id in range(self._num_replicas):
            replica = self.shared_state.replica_budgets[replica_id]
            replica_number_pending_tokens = self.replica_slo_budget - replica.current_budget
            global_waiting_tokens = self.global_request_queue.get_global_actual_waiting_tokens_count(replica_id)
            number_pending_tokens_list.append(replica_number_pending_tokens + global_waiting_tokens)
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
            for replica_id in range(self._num_replicas):
                replica = self.shared_state.replica_budgets[replica_id]
                replica_input_pending_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
                global_input_waiting_tokens = self.global_request_queue.get_global_input_waiting_tokens_count(replica_id)
                number_input_tokens_list.append(replica_input_pending_tokens + global_input_waiting_tokens)

            os.makedirs(base_path, exist_ok=True)
            try:
                with open(f'{base_path}/number_pending_input_tokens.log', "a+") as file:
                    for replica_id in range(len(number_input_tokens_list)):
                        file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_input_tokens,{number_input_tokens_list[replica_id]}\n')
            except Exception as e:
                print(f'error:MetricsConfig:save number_input_tokens failed! {e}')
