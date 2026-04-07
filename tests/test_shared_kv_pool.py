from types import SimpleNamespace

from dualmap.cache_manager.kvcache_store.kvcache_engine import KvCacheEngine
from dualmap.cache_manager.kvcache_store.shared_prefix_kvcache_pool import SharedPrefixKvCachePool
from dualmap.entities.replica import Replica
from dualmap.entities.request import Request
from dualmap.scheduler.utils.double_hash_global_scheduler_utils import DoubleHashGlobalSchedulerUtils
from dualmap.scheduler.utils.shared import SharedState


class _DummyTokenizer:
    def encode(self, text):
        return [1, 2, 3]


class _DummyMetricStore:
    async def insert_metrics(self, data):
        return


def _args(enable_shared=True):
    return SimpleNamespace(
        replica_slo_budget=10000,
        cache_capacity=1024 * 1024,
        block_size=4,
        kv_cache_size_per_token=1,
        enable_shared_kv_pool=enable_shared,
        enable_cross_instance_reuse=True,
        replicas_ip_port="127.0.0.1:9001,127.0.0.1:9002",
        prefill_tpot=1.0,
        tpct=1.0,
        tprt=2.0,
        local_hit_tprt=0.1,
        remote_hit_tprt=0.5,
        result_path="/tmp",
        model_name="test-model",
        balance_type="dualmap_min_ttft",
        dh_first_balance_ttft_thredhold=1,
        dh_rebalance_thredhold=1,
        dh_replica_pending_req_threshold=8,
        dh_rebalance_waiting_latency_thredhold=100,
        dh_recompute_punish_ratio=0,
        busy_prefill_interval=100,
    )


def _req(req_id=1, token_count=16):
    input_ids = list(range(token_count))
    return Request(
        request_id=req_id,
        dataset_type="test",
        native_session_id=1,
        session_id="s",
        hash_session_id="s",
        round_id=1,
        prompts="hello",
        input_ids=input_ids,
        num_prefill_tokens=len(input_ids),
        actual_num_prefill_tokens=len(input_ids),
        output_len=8,
        over_flow=False,
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=8,
        stream=True,
        arrived_at=0.0,
        time_interval=0.0,
        hash_prefix_len=0,
    )


def _req_with_input(req_id, input_ids):
    req = _req(req_id=req_id, token_count=len(input_ids))
    req._input_ids = input_ids
    req._num_prefill_tokens = len(input_ids)
    return req


def test_cross_instance_reuse_tokens_are_reusable_not_miss():
    args = _args(enable_shared=True)
    pool = SharedPrefixKvCachePool(KvCacheEngine(args.cache_capacity, args.block_size, args.kv_cache_size_per_token))
    r0 = Replica(0, _DummyTokenizer(), args, shared_cache_pool=pool)
    r1 = Replica(1, _DummyTokenizer(), args, shared_cache_pool=pool)

    token_ids = list(range(16))
    r0.save_prefill_token_ids("r0", token_ids)

    # Replica1 sees cross-instance reusable tokens.
    assert r1.get_num_remote_hit_token_ids(token_ids) == len(token_ids)
    assert r1.get_num_recompute_token_ids(token_ids) == 0


def test_cross_instance_reuse_excluded_from_full_recompute_on_admission():
    args = _args(enable_shared=True)
    pool = SharedPrefixKvCachePool(KvCacheEngine(args.cache_capacity, args.block_size, args.kv_cache_size_per_token))
    r0 = Replica(0, _DummyTokenizer(), args, shared_cache_pool=pool)
    r1 = Replica(1, _DummyTokenizer(), args, shared_cache_pool=pool)

    req = _req(req_id=42)
    r0.save_prefill_token_ids("prime", req._input_ids)

    import asyncio

    async def _run():
        await r1.add_request(req)

    asyncio.run(_run())
    assert req._actual_num_prefill_tokens == 0
    assert req._remote_hit_tokens == len(req._input_ids)


def test_ttft_distinguishes_local_vs_cross_instance_reuse_costs():
    args = _args(enable_shared=True)
    state = SharedState(metric_store=_DummyMetricStore(), tokenizer=_DummyTokenizer(), args=args)

    req_prime = _req(req_id=10)
    state.replica_budgets[0].save_prefill_token_ids(req_prime._id, req_prime._input_ids)
    req = _req(req_id=11)

    import asyncio

    async def _run():
        rid, actual_prefill = await state.get_min_ttft_replica(req)
        return rid, actual_prefill

    rid, actual_prefill = asyncio.run(_run())
    # replica0 has local reuse, replica1 only cross-instance reuse; local cost is lower.
    assert rid == 0
    assert actual_prefill == 0


def test_double_hash_estimator_distinguishes_local_vs_cross_instance_costs():
    args = _args(enable_shared=True)
    dummy_shared_state = SimpleNamespace()
    util = DoubleHashGlobalSchedulerUtils(num_replicas=2, shared_state=dummy_shared_state, args=args)

    # Same token volume, but local reuse should be cheaper than cross-instance reuse.
    local_cost = util.estimate_ttft(waiting_tokens=0, waiting_hit_tokens=16, hit_tokens=0, waiting_local_hit_tokens=16, local_hit_tokens=0)
    cross_cost = util.estimate_ttft(waiting_tokens=0, waiting_hit_tokens=16, hit_tokens=0, waiting_local_hit_tokens=0, local_hit_tokens=0)
    assert local_cost < cross_cost


def test_double_hash_pending_ttft_uses_local_pending_hit_cost():
    args = _args(enable_shared=True)

    class _FakeReplica:
        def get_num_pending_hit_tokens(self):
            return 10

        def get_num_pending_local_hit_tokens(self):
            return 10

        async def get_load_states(self):
            return 10000, 0.0, 0.0, 0, 0.0

        def get_num_running_req(self):
            return 0

        def get_running_req_blocks_cnt(self):
            return 0

    fake_shared_state = SimpleNamespace(
        replica_budgets={0: _FakeReplica()},
        get_num_actual_pending_tokens_replica=lambda rid: 0,
        get_pending_input_tokens_replica=lambda rid: 0,
    )
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=fake_shared_state, args=args)
    util.global_request_queue.queues_global_waiting_hit_tokens_count[0] = 0

    rebalanced = {"called": False}

    async def _fake_rebalance(*_args, **_kwargs):
        rebalanced["called"] = True

    util.rebalance_replica_global_waiting_reqs = _fake_rebalance

    import asyncio

    asyncio.run(util.get_schedulable_waiting_req_list())
    # Local pending hits should use local_hit_tprt and avoid false overload rebalance.
    assert rebalanced["called"] is False


def test_feature_flag_off_preserves_local_only_behavior():
    args = _args(enable_shared=False)
    r0 = Replica(0, _DummyTokenizer(), args)
    r1 = Replica(1, _DummyTokenizer(), args)
    token_ids = list(range(16))

    r0.save_prefill_token_ids("r0", token_ids)
    assert r1.get_num_recompute_token_ids(token_ids) == len(token_ids)


def test_lightweight_integration_request_lifecycle_for_cross_instance_reuse():
    args = _args(enable_shared=True)
    state = SharedState(metric_store=_DummyMetricStore(), tokenizer=_DummyTokenizer(), args=args)

    shared_prompt = list(range(16))
    miss_prompt = list(range(100, 116))
    req_prime = _req_with_input(req_id=201, input_ids=shared_prompt)
    state.replica_budgets[0].save_prefill_token_ids(req_prime._id, req_prime._input_ids)

    # Scheduling cost chain: local reuse should be cheaper than cross-instance reuse.
    req_for_sched = _req_with_input(req_id=202, input_ids=shared_prompt)
    import asyncio

    async def _pick_replica():
        return await state.get_min_ttft_replica(req_for_sched)

    chosen_replica, _ = asyncio.run(_pick_replica())
    assert chosen_replica == 0

    # Request admission on replica-1: cross-instance reusable tokens must reduce recompute.
    req_cross = _req_with_input(req_id=203, input_ids=shared_prompt)

    async def _admit_cross():
        await state.replica_budgets[1].add_request(req_cross)

    asyncio.run(_admit_cross())
    assert req_cross._local_hit_tokens == 0
    assert req_cross._remote_hit_tokens == len(shared_prompt)
    assert req_cross._actual_num_prefill_tokens == 0

    # Miss path on replica-1 should still be full recompute.
    req_miss = _req_with_input(req_id=204, input_ids=miss_prompt)

    async def _admit_miss():
        await state.replica_budgets[1].add_request(req_miss)

    asyncio.run(_admit_miss())
    assert req_miss._remote_hit_tokens == 0
    assert req_miss._actual_num_prefill_tokens == len(miss_prompt)

    # Backward compatibility: shared pool off falls back to local-only behavior.
    args_off = _args(enable_shared=False)
    state_off = SharedState(metric_store=_DummyMetricStore(), tokenizer=_DummyTokenizer(), args=args_off)
    state_off.replica_budgets[0].save_prefill_token_ids("off_prime", shared_prompt)
    req_off = _req_with_input(req_id=205, input_ids=shared_prompt)

    async def _admit_off():
        await state_off.replica_budgets[1].add_request(req_off)

    asyncio.run(_admit_off())
    assert req_off._actual_num_prefill_tokens == len(shared_prompt)
