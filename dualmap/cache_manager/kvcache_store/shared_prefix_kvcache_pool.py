from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PrefixCacheView:
    # cached on current replica and directly reusable at local-cache cost
    total_cached_blocks: int
    # subset of total_cached_blocks located on current replica
    local_cached_blocks: int
    # subset of total_cached_blocks located on other replicas but reusable
    # through shared-pool cross-instance semantics in this codebase
    remote_cached_blocks: int
    # not found in shared pool and therefore must be prefetched/recomputed
    missing_blocks: int


class SharedPrefixKvCachePool:
    """Global KV cache pool with a shared namespace across replicas.

    This models a unified cache namespace where each prefix block hash may be
    materialized on multiple replicas, while lifecycle (pin/refcount + eviction)
    is coordinated globally.
    """

    def __init__(self, cache_engine):
        self.cache_engine = cache_engine
        self.block_size = cache_engine.block_size
        self.num_blocks = cache_engine.get_num_blocks()
        self._min_watermark = int(0.01 * self.num_blocks)

        # block_hash -> global metadata
        self._global_blocks: Dict[int, Dict[str, object]] = {}
        # request_id -> pinned block hashes
        self._request_pins: Dict[object, Set[int]] = defaultdict(set)
        # Global LRU of block hashes
        self._global_lru = OrderedDict()

    def _iter_prefix_hashes(self, prompt_token_ids: List[int]):
        is_first_block = True
        prev_block_hash = None
        block_num = len(prompt_token_ids) // self.block_size
        for logic_id in range(block_num):
            cur_block_token_ids = prompt_token_ids[logic_id * self.block_size : (logic_id + 1) * self.block_size]
            block_hash = hash((is_first_block, prev_block_hash, *cur_block_token_ids))
            yield block_hash
            prev_block_hash = block_hash
            is_first_block = False

    def query_prefix(self, replica_id: int, prompt_token_ids: List[int]) -> PrefixCacheView:
        total = local = remote = 0
        for block_hash in self._iter_prefix_hashes(prompt_token_ids):
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                break
            total += 1
            locations = entry["locations"]
            if replica_id in locations:
                local += 1
            else:
                remote += 1
            self._touch(block_hash)
        block_num = len(prompt_token_ids) // self.block_size
        return PrefixCacheView(
            total_cached_blocks=total,
            local_cached_blocks=local,
            remote_cached_blocks=remote,
            missing_blocks=max(0, block_num - total),
        )

    def ensure_materialized(self, replica_id: int, prompt_token_ids: List[int]) -> int:
        """Materialize globally cached blocks onto replica; returns transferred blocks."""
        transferred = 0
        for block_hash in self._iter_prefix_hashes(prompt_token_ids):
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                break
            locations = entry["locations"]
            if replica_id not in locations:
                locations.add(replica_id)
                transferred += 1
            self._touch(block_hash)
        return transferred

    def pin_request_prefix(self, request_id: object, prompt_token_ids: List[int], replica_id: Optional[int] = None, include_remote: bool = True) -> None:
        for block_hash in self._iter_prefix_hashes(prompt_token_ids):
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                break
            if replica_id is not None and not include_remote and replica_id not in entry["locations"]:
                continue
            if block_hash not in self._request_pins[request_id]:
                entry["pin_count"] += 1
                self._request_pins[request_id].add(block_hash)

    def unpin_request(self, request_id: object) -> None:
        pinned = self._request_pins.pop(request_id, set())
        for block_hash in pinned:
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                continue
            entry["pin_count"] = max(0, entry["pin_count"] - 1)

    def save_prefill(self, replica_id: int, prompt_token_ids: List[int]) -> int:
        num_saved_blocks = 0
        for block_hash in self._iter_prefix_hashes(prompt_token_ids):
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                self._allocate_block(block_hash, replica_id)
                num_saved_blocks += 1
            else:
                entry["locations"].add(replica_id)
                self._touch(block_hash)
        return num_saved_blocks

    def save_decode(self, replica_id: int, token_ids: List[int]) -> int:
        return self.save_prefill(replica_id, token_ids)

    def _allocate_block(self, block_hash: int, replica_id: int):
        self._check_and_reclaim_cache()
        if len(self._global_blocks) >= self.num_blocks:
            self._evict(1)
        if len(self._global_blocks) >= self.num_blocks:
            return

        self._global_blocks[block_hash] = {
            "locations": {replica_id},
            "pin_count": 0,
        }
        self._touch(block_hash)

    def _touch(self, block_hash: int):
        if block_hash in self._global_lru:
            self._global_lru.move_to_end(block_hash)
        else:
            self._global_lru[block_hash] = True

    def _check_and_reclaim_cache(self):
        free_blocks = self.num_blocks - len(self._global_blocks)
        if free_blocks < self._min_watermark:
            self._evict(min(80, max(1, self.num_blocks // 32)))

    def _evict(self, num_blocks: int):
        evicted = 0
        while evicted < num_blocks and self._global_lru:
            block_hash, _ = self._global_lru.popitem(last=False)
            entry = self._global_blocks.get(block_hash)
            if entry is None:
                continue
            if entry["pin_count"] > 0:
                self._global_lru[block_hash] = True
                continue
            self._global_blocks.pop(block_hash, None)
            evicted += 1

    def get_size(self):
        return len(self._global_blocks)

    def get_global_block_state(self) -> Dict[int, Tuple[Set[int], int]]:
        return {
            block_hash: (set(meta["locations"]), int(meta["pin_count"]))
            for block_hash, meta in self._global_blocks.items()
        }
