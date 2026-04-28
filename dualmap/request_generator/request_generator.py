import asyncio
import time
import json
import os
from datetime import datetime
from dualmap.scheduler.utils.shared import SharedState
from dualmap.scheduler.utils.lazy_prefix_table import LazyPrefixTable,HotPrefixDetector,LazyExpansionController
from itertools import count
from dualmap.entities.request import Request
from dualmap.logger import init_logger

logger = init_logger(__name__)
request_id_counter = count()

class RequestGenerator:
    def __init__(self, shared_state: SharedState, tokenizer, num_replicas, args):
        self.shared_state = shared_state
        self._args = args
        self._num_replicas = num_replicas
        self._max_model_len = args.max_model_len
        self._dataset_type = args.dataset_type
        self._request_dataset_dir = args.request_dataset_dir
        self._dataset_file = args.dataset_file
        self._request_generate_qps = float(args.request_generate_qps)
        self._num_request = args.request_num
        self._warm_up_requests_num = args.warm_up_requests_num
        self._warm_up_qps = float(getattr(args, "warm_up_qps", 0.5) or 0.5)
        self._requests_num_dataset_start = args.requests_num_dataset_start
        self._cur_num_request = 0
        self._active_timeout = args.request_active_timeout  #s
        self._is_finished = False
        self.prefix_table = LazyPrefixTable()
        self.hot_prefix_detector = HotPrefixDetector()
        self.prefix_expansion_ctrl = LazyExpansionController(self.prefix_table, self.hot_prefix_detector)   

    def is_request_active(self) -> bool:
        last_request_time = self.shared_state.last_request_time
        if last_request_time is None:
            return True
        if self._is_finished:
            return False
        if not last_request_time:
            last_request_time = time.perf_counter()
            self.shared_state.last_request_time = time.perf_counter() 
        return time.perf_counter() - last_request_time < self._active_timeout

    async def _generate_request_helper(self, request: Request, native_session_id: str):
        # logger.info(f"_generate_request_helper: {request._id}, {request._hash_session_id}")
        if request is None:
            return
        start = time.perf_counter()
        event = asyncio.Event()
        self.shared_state.runtime_events[request._id] = (event, None)
        await self.shared_state.runtime_request_queue.put((request))

        start = time.perf_counter()
        await self.shared_state.runtime_events[request._id][0].wait()
        replica_id = self.shared_state.runtime_events[request._id][1]
        self.shared_state.runtime_events.pop(request._id)

        if replica_id is None:
            raise RuntimeError("Runtime selection failed")
        if replica_id < 0:
           return 
        if replica_id >= 0 and replica_id < self._num_replicas:
            await self.shared_state.add_posting_request_tasks(replica_id, request)

    async def generate_request_offline(self, record: dict, time_interval, dataset_type) -> dict:
        prompts = record["prompts"]
        input_ids = record["input_ids"]
        output_len = record["output_len"]
        if getattr(self._args, "force_output_len_1", False):
            output_len = 1
        g_session_id = record["g_session_id"]
        hash_session_id = record["hash_session_id"]
        hash_ids = record["hash_ids"]
        if g_session_id == "" or hash_session_id == ""\
            or prompts == "" or len(input_ids) > self._max_model_len or len(input_ids) == 0\
            or output_len <= 0 or hash_ids is None or len(hash_ids) < 1:
            return

        parts_hash_session_id = hash_session_id.split("@")
        if len(parts_hash_session_id) < 2:
            return
        session_id = parts_hash_session_id[1]

        if len(input_ids) >= self._max_model_len:
            return

        parts_g_session_id = g_session_id.split("@")
        if len(parts_g_session_id) < 3:
            return
        hash_prefix_len = self.prefix_table.lookup(hash_ids)
        self.prefix_expansion_ctrl.process(hash_ids)
        hash_prefix_str = "".join(map(str, hash_ids[:hash_prefix_len]))
        request_id = next(request_id_counter)
        new_g_session_id = f"{parts_g_session_id[0]}@{parts_g_session_id[1]}@{request_id}"
        hash_session_id = f"{parts_hash_session_id[0]}@{hash_prefix_str}"
        logger.info(f"Generating request: {request_id}, {new_g_session_id}, {hash_session_id}, {hash_prefix_len}")
        request = Request(
            request_id = int(request_id),
            dataset_type = dataset_type,
            native_session_id = int(session_id),
            session_id = new_g_session_id,
            hash_session_id = hash_session_id,
            round_id = 0, 
            prompts = prompts,
            input_ids = input_ids,
            num_prefill_tokens = len(input_ids),
            actual_num_prefill_tokens = len(input_ids),
            output_len = int(output_len),
            over_flow = False,
            n = 1,
            temperature = 0,
            top_p = 1,
            max_tokens = int(output_len),
            stream = True,
            arrived_at = time.perf_counter(),
            time_interval = time_interval,
            hash_prefix_len = hash_prefix_len
        )
        self._cur_num_request += 1
        await self._generate_request_helper(request, session_id)

    def valid_record_offline(self, record):
        valid = True
        if record["timestamp"] < 0:
            valid = False
        return valid

    async def generate_from_file(self):   
        dataset_file = self._dataset_file
        request_native_qps = 0
        if "conversation" in self._dataset_type:
            request_native_qps = 3.34
            self._warm_up_qps = 0.3 
        elif "toolagent" in self._dataset_type:
            request_native_qps = 6.5 / 2 
            self._warm_up_qps = 0.5 

        time_interval = 1
        logger.info(f"Generating requests from file: {dataset_file}")
        try:    
            with open(dataset_file, 'r') as file:
                logger.info(f"Reading file: {dataset_file}")
                current_group = []
                prev_timestamp = None

                cur_line_num = 0
                while True:
                    if self._cur_num_request >= self._num_request:
                        break    

                    line = file.readline()
                    cur_line_num += 1
                    if self._requests_num_dataset_start >= cur_line_num:
                        continue

                    if not line:  
                        for record_in_group in current_group:
                            start = time.perf_counter()
                            await self.generate_request_offline(record_in_group, time_interval, self._dataset_type)
                            delay = time.perf_counter() - start
                            await asyncio.sleep(max(0, time_interval - delay))
                            await asyncio.sleep(0)                    
                        break
                    
                    record = json.loads(line)
                    if not self.valid_record_offline(record):
                        continue

                    current_timestamp = record["timestamp"]
                    
                    if prev_timestamp is None:
                        prev_timestamp = current_timestamp               
                    
                    if current_timestamp == prev_timestamp:
                        current_group.append(record)
                    else:
                        if current_group:
                            native_time_interval = round((current_timestamp - prev_timestamp) / len(current_group) / 1000, 3)
                            if self._warm_up_requests_num > self._cur_num_request:
                                if self._warm_up_qps > 0:
                                    time_interval = 1/self._warm_up_qps # warm up tool-agent:qps=0.5, mooncake-conversation qps=0.2
                                else:
                                    time_interval = 2
                            else:
                                time_interval = round(native_time_interval * request_native_qps / self._request_generate_qps, 3)
                            for record_in_group in current_group:
                                start = time.perf_counter()
                                await self.generate_request_offline(record_in_group, time_interval, self._dataset_type)
                                delay = time.perf_counter() - start
                                await asyncio.sleep(max(0, time_interval - delay)) 
                                await asyncio.sleep(0) 
                        current_group = [record]
                        prev_timestamp = current_timestamp
        
        except Exception as e:
            logger.error(f"Error generating requests from file: {dataset_file}, error: {e}")
            return
