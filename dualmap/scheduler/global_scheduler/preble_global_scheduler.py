from datetime import datetime
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.preble_global_scheduler_utils import PrebleGlobalSchedulerUtils
from dualmap.scheduler.utils.shared import SharedState

from dualmap.logger import init_logger

logger = init_logger(__name__)

class PrebleGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, num_replicas, window_duration: int, shared_state: SharedState, args):
        super().__init__(num_replicas)
        self.preble_schedule_util = PrebleGlobalSchedulerUtils(num_replicas, window_duration, args)
        self.shared_state = shared_state

    async def schedule(self, request: Request) -> int:
        if request is None:
            return -1
        # Tokenize the text
        runtime_id_with_highest_hit_rate = None # None in origin 
        decoding_length = request._output_len
        runtime_idx = -1
        with self.preble_schedule_util.lock:
            split_nodes = {}   
            leaf_node = self.preble_schedule_util.cache.insert(tuple(request._input_ids), split_nodes=split_nodes) 
            self.preble_schedule_util.update_gpu_allocations_with_split_nodes(split_nodes, self.preble_schedule_util.gpu_allocations) # copies split node gpu allocation
            self.preble_schedule_util.update_histogram_with_split_node(split_nodes)
            nearest_ancestor_node_with_low_share_prefix_ratio = self.preble_schedule_util.get_nearest_ancestor_node_with_low_share_prefix_ratio(leaf_node)
            if leaf_node.num_tokens < leaf_node.context_so_far: # num_recaculate_tokens < num_share_prefix_tokens
                gpu_selected = self.preble_schedule_util.get_parent_gpu_allocation(leaf_node)
                if len(gpu_selected) > 1:
                    logger.debug(f'schedule len(gpu_selected) > 1: gpu_selected-{gpu_selected}')
                    runtime_idx = self.preble_schedule_util.calculate_min_load_cost(leaf_node, gpu_selected)
                else:
                    runtime_idx = list(gpu_selected)[0]
            elif runtime_id_with_highest_hit_rate is not None:
                runtime_idx = runtime_id_with_highest_hit_rate
            else:
                runtime_idx = self.preble_schedule_util.calculate_min_load_cost(leaf_node, selected_gpus=range(self.preble_schedule_util.num_gpus))
            self.preble_schedule_util.counter += 1

            if runtime_idx == -1:
                logger.debug("debug: runtime_idx == -1")
            assert runtime_idx != -1


            if leaf_node.context_length > leaf_node.num_tokens:
                logger.debug(f"debug:PrebleGlobalScheduler:schedule cache hit: replica-{runtime_idx}, session-{request._session_id},round-{request._round_id},leaf_node.context_so_far:{leaf_node.context_so_far}, leaf_node.num_tokens:{leaf_node.num_tokens}")

            self.preble_schedule_util.add_gpu_allocation_for_parent(leaf_node, {runtime_idx}) # Updated gpu allocations up till parent
            self.preble_schedule_util.cache.update_cache_metadada(leaf_node, runtime_idx) # Update ref counters

            assert self.preble_schedule_util.is_unshare_prefix_over_50_percent_for_node(nearest_ancestor_node_with_low_share_prefix_ratio)

            self.preble_schedule_util.histogram.add_request(datetime.now(),nearest_ancestor_node_with_low_share_prefix_ratio, leaf_node, runtime_idx, decoding_length=decoding_length)
            self.preble_schedule_util.num_requests_for_gpus_list[runtime_idx] += 1

            # NOTE: eviction handled by iterative feedback
            if self.preble_schedule_util.enable_eviction:
                self.preble_schedule_util.handle_cache_eviction_for_gpu(runtime_idx) 
            if self.preble_schedule_util.enable_rebalancing:
    
                if leaf_node.depth - nearest_ancestor_node_with_low_share_prefix_ratio.depth < self.preble_schedule_util.REBALANCING_CHAIN_LENGTH: # Ignore longer chains for Infercept optimizations
                    self.preble_schedule_util.handle_important_node_stealing(runtime_idx)
        return runtime_idx
    
    def finish_request(
        self, func_output: RequestFuncOutput=None, text: str = None, input_ids=None
    ):
        with self.preble_schedule_util.lock:
            if func_output is None or not func_output.success:
                return
            runtime_id = func_output.runtime_selected
            self.preble_schedule_util.update_overload_detector(input_ids, runtime_id, func_output)
            important_node = self.preble_schedule_util.get_nearest_ancestor_node_with_low_share_prefix_ratio(self.preble_schedule_util.cache.find_node(input_ids))
            self.preble_schedule_util.cache.remove_completed_input_ids(input_ids, runtime_id)
            if func_output.tpot != 0 and func_output.output_len != 1:
                self.preble_schedule_util.topt_queues_for_gpus[runtime_id].append(func_output.tpot)
            self.preble_schedule_util.histogram.current_decode_lengths_per_gpu[runtime_id] -= func_output.max_new_tokens
            self.preble_schedule_util.histogram.total_decode_length_for_node_maps[important_node] -= func_output.max_new_tokens