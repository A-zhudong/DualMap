from typing import List
from dualmap.cache_manager.kvcache_store.kvcache_engine import KvCacheEngine
from dualmap.cache_manager.kvcache_store.prefix_kvcache_lru import PrefixLRUCache
from dualmap.logger import init_logger
from typing import Dict, List, Optional, Set
BlockId = int
PrefixHash = int

logger = init_logger(__name__)

class PrefixKVCacheMgr:
    def __init__(
        self, 
        cache_engine: KvCacheEngine,
    ):
        self.cache_engine = cache_engine
        self.num_blocks = self.cache_engine.get_num_blocks()
        self.lru = PrefixLRUCache(self.num_blocks)
        self._free_block_indices: Set[BlockId] = set(range(self.num_blocks))
        self._min_watermark = int(0.01 * self.num_blocks)
        self._cached_blocks: Dict[PrefixHash, BlockId] = {}

    def get_num_cached_blocks(self, prompt_token_ids: List[int]) -> int:
        block_size = self.cache_engine.block_size
        is_first_block = True
        prev_block_hash = None
        last_commit_block_id = -1
        block_num = len(prompt_token_ids) // block_size
        for logic_id in range(block_num):
            assert (prev_block_hash is None) == is_first_block
            cur_block_token_ids = prompt_token_ids[logic_id * block_size : (logic_id + 1) * block_size]
            block_hash = hash((is_first_block, prev_block_hash, *cur_block_token_ids))
            index = self._cached_blocks.get(block_hash, None)
            res = self.lru.query(block_hash)
            if index is None or res == -1:
                break
            else:
                last_commit_block_id += 1
                prev_block_hash = block_hash
                is_first_block = False
        num_cached_blocks = last_commit_block_id + 1
        return num_cached_blocks 

    def save(self, prompt_token_ids: List[int]) -> int:
        num_saved_blocks = self._save(prompt_token_ids)
        return num_saved_blocks


    def save_prefill(self, request_id, prompt_token_ids: List[int]) -> int:
        num_saved_blocks = self._save_prefill(request_id, prompt_token_ids)
        return num_saved_blocks

    def _save(self, prompt_token_ids: List[int]):
        index = -1
        block_size = self.cache_engine.block_size
        is_first_block = True
        prev_block_hash = None
        num_saved_blocks = 0
        block_num = len(prompt_token_ids) // block_size
        for logic_id in range(block_num):
            assert (prev_block_hash is None) == is_first_block
            cur_block_token_ids = prompt_token_ids[logic_id * block_size : (logic_id + 1) * block_size]
            block_hash = hash((is_first_block, prev_block_hash, *cur_block_token_ids))
            index = self._cached_blocks.get(block_hash, None)
            res = self.lru.get(block_hash)
            if index is None:
                try:
                    self._allocate_block(block_hash)
                except Exception:
                    pass  
                self.lru.put(block_hash, 0)
                num_saved_blocks += 1
            prev_block_hash = block_hash
            is_first_block = False
        return num_saved_blocks


    def _save_prefill(self, request_id, prompt_token_ids: List[int]):
        index = -1
        block_size = self.cache_engine.block_size
        is_first_block = True
        prev_block_hash = None
        num_saved_blocks = 0
        block_num = len(prompt_token_ids) // block_size
        for logic_id in range(block_num):
            assert (prev_block_hash is None) == is_first_block
            cur_block_token_ids = prompt_token_ids[logic_id * block_size : (logic_id + 1) * block_size]
            block_hash = hash((is_first_block, prev_block_hash, *cur_block_token_ids))
            index = self._cached_blocks.get(block_hash, None)
            res = self.lru.get(block_hash)
            if index is None:
                try:
                    self._allocate_block(block_hash)
                except Exception:
                    pass  
                self.lru.put(block_hash, 0)

                num_saved_blocks += 1
            prev_block_hash = block_hash
            is_first_block = False
        return num_saved_blocks


    def _allocate_block(self, block_hash: Optional[int]):
        block_id = -1
        try:
            block_id = self._allocate_new_block_id()
            assert block_id is not None
            if block_hash not in self._cached_blocks:
                self._cached_blocks[block_hash] = block_id
            else: 
                self._free_block(block_hash)
        except Exception:
            pass
        return block_id


    def _free_block(self, content_hash_to_evict: PrefixHash) -> bool:
        success = False
        if content_hash_to_evict is None or content_hash_to_evict not in self._cached_blocks:
            return success
        
        block_id = self._cached_blocks[content_hash_to_evict]
        self._cached_blocks.pop(content_hash_to_evict)
        self._free_block_id(block_id)
        success = True
        return success

    def _allocate_new_block_id(self) -> BlockId:
        block_id = -1
        self._check_and_reclaim_cache()
        if not self._free_block_indices:
            return block_id
        block_id = next(iter(self._free_block_indices))
        self._free_block_indices.remove(block_id)
        return block_id

    def _free_block_id(self, block_id: BlockId) -> None:
        self._free_block_indices.add(block_id)

    def _evict(self, num_evict_blocks: int):
        num_evicted_blocks = 0
        while(num_evict_blocks >= 0):
            block_hash, _ = self.lru.pop()
            if not block_hash:
                logger.error(f"DEBUG:[evict]lru is empty!!!!!!")
            else:
                success = self._free_block(block_hash)
                if success:
                    num_evict_blocks -= 1
                    num_evicted_blocks += 1
        logger.warning(f"DEBUG:[evict]delete num_evicted_blocks:{num_evicted_blocks}")

    def _check_and_reclaim_cache(self):
        if self.get_num_free_blocks() < self._min_watermark:
            num_evict_blocks = 80 # 10240 tokens
            self._evict(num_evict_blocks)
        return

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

    def get_size(self):
        return self.lru.get_len()