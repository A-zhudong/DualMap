from typing import List, Optional

class Request():
    def __init__(
        self,
        request_id: int,
        dataset_type: str,
        native_session_id: int,
        session_id: str,
        hash_session_id: str,
        round_id: int, 
        prompts: str,
        input_ids: Optional[List[int]],
        num_prefill_tokens: int,
        actual_num_prefill_tokens: int,
        output_len: int,
        over_flow: bool,
        n: int,
        temperature: int,
        top_p: int,
        max_tokens: int,
        stream:bool,
        arrived_at: float,
        time_interval:float,
        hash_prefix_len:int
    ):
        self._id = request_id
        self._dataset_type = dataset_type
        self._native_session_id = native_session_id
        self._session_id = session_id
        self._hash_session_id = hash_session_id
        self._round_id = round_id
        self._prompts = prompts
        self._input_ids = input_ids
        self._num_prefill_tokens = num_prefill_tokens # len(self._prefill_tokens)
        self._actual_num_prefill_tokens = actual_num_prefill_tokens  
        self._output_len = output_len
        self._over_flow = over_flow
        self._n= n
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._stream = stream
        self._arrived_at = arrived_at
        self._time_interval = time_interval
        self._primary_replica = -1 # for double_hash
        self._second_replica = -1 # for double_hash
        self._rounting_cache_hit_max = -1 # for double_hash
        self._is_dh_cache_affinity = 0 # for double_hash
        self._is_dh_least_loaded = 0 # for double_hash
        self._is_dh_cache_affinity_least_loaded = 0 # for double_hash
        self._hash_prefix_len = hash_prefix_len