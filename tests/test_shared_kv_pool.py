from types import SimpleNamespace

from dualmap.cache_manager.kvcache_store.kvcache_engine import KvCacheEngine
from dualmap.cache_manager.kvcache_store.shared_prefix_kvcache_pool import SharedPrefixKvCachePool
from dualmap.entities.replica import Replica
from dualmap.entities.request import Request
from dualmap.scheduler.utils.double_hash_global_scheduler_utils import DoubleHashGlobalSchedulerUtils, GlobalRequestQueue
from dualmap.scheduler.utils.shared import SharedState
from dualmap.client.open_ai import get_prefix_kv_load_priority


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
        priority_enabled=True,
        priority_policy="quantile",
        priority_quantile=0.99,
        priority_window_size=256,
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


def _req_with_arrival(req_id, arrival, token_count=8):
    req = _req(req_id=req_id, token_count=token_count)
    req._arrived_at = arrival
    return req


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


def test_pr1_global_queue_order_priority_then_cache_hit_then_arrival():
    queue = GlobalRequestQueue(num_replicas=1)
    high_low_hit = _req_with_arrival(req_id=301, arrival=3.0)
    setattr(high_low_hit, "_priority_rank", 0)
    low_high_hit = _req_with_arrival(req_id=302, arrival=1.0)
    setattr(low_high_hit, "_priority_rank", 1)
    high_high_hit_early = _req_with_arrival(req_id=303, arrival=1.0)
    setattr(high_high_hit_early, "_priority_rank", 0)
    high_high_hit_late = _req_with_arrival(req_id=304, arrival=2.0)
    setattr(high_high_hit_late, "_priority_rank", 0)

    queue.push(0, high_low_hit, prefix_cache_hit_len=1)
    queue.push(0, low_high_hit, prefix_cache_hit_len=99)
    queue.push(0, high_high_hit_late, prefix_cache_hit_len=2)
    queue.push(0, high_high_hit_early, prefix_cache_hit_len=2)

    assert queue.pop(0)._id == 303  # same priority/hit -> earlier arrival
    assert queue.pop(0)._id == 304
    assert queue.pop(0)._id == 301  # high priority beats low priority
    assert queue.pop(0)._id == 302


def test_pr1_add_request_sets_predicted_ttft_and_priority():
    args = _args(enable_shared=True)
    args.balance_type = "dualmap_min_ttft"
    args.dh_first_balance_ttft_thredhold = 10

    class _FakeReplica:
        def __init__(self, recompute_tokens):
            self.recompute_tokens = recompute_tokens

        def get_num_pending_hit_tokens(self):
            return 0

        def get_num_pending_local_hit_tokens(self):
            return 0

        def get_num_recompute_token_ids(self, _):
            return self.recompute_tokens

        def get_num_hit_token_ids(self, input_ids):
            return len(input_ids) - self.recompute_tokens

        def get_num_local_hit_token_ids(self, _):
            return 0

        async def get_load_states(self):
            return 10000, 0.0, 0.0, 0, 0.0

        def get_num_running_req(self):
            return 0

        def get_running_req_blocks_cnt(self):
            return 0

    fake_state = SimpleNamespace(
        replica_budgets={0: _FakeReplica(1), 1: _FakeReplica(8)},
        get_num_actual_pending_tokens_replica=lambda rid: 0,
        get_pending_input_tokens_replica=lambda rid: 0,
        dump_replica_queue_info=lambda rid: None,
    )
    util = DoubleHashGlobalSchedulerUtils(num_replicas=2, shared_state=fake_state, args=args)
    req = _req(req_id=311, token_count=8)

    import asyncio

    asyncio.run(util.add_request_to_best_global_queue(req))

    assert hasattr(req, "_estimated_ttft")
    assert req._priority_level in {"HIGH", "LOW"}
    assert req._priority_rank in {0, 1}
    assert req._priority_level == "LOW"


def test_pr1_rebalance_recomputes_priority_on_migration():
    args = _args(enable_shared=True)
    args.balance_type = "dualmap"
    args.dh_first_balance_ttft_thredhold = 5
    args.dh_rebalance_thredhold = 100

    class _FakeReplica:
        def __init__(self, rid):
            self.rid = rid

        def get_num_pending_hit_tokens(self):
            return 0

        def get_num_pending_local_hit_tokens(self):
            return 0

        def get_num_recompute_token_ids(self, input_ids):
            # source (rid=0) is expensive; target (rid=1) is cheap due hits
            return len(input_ids) if self.rid == 0 else 0

        def get_num_hit_token_ids(self, input_ids):
            return 0 if self.rid == 0 else len(input_ids)

        async def get_load_states(self):
            return 10000, 0.0, 0.0, 0, 0.0

    fake_state = SimpleNamespace(
        replica_budgets={0: _FakeReplica(0), 1: _FakeReplica(1)},
        get_num_actual_pending_tokens_replica=lambda rid: 0,
        dump_replica_queue_info=lambda rid: None,
    )
    util = DoubleHashGlobalSchedulerUtils(num_replicas=2, shared_state=fake_state, args=args)

    req_a = _req(req_id=321, token_count=8)
    req_b = _req(req_id=322, token_count=8)
    setattr(req_a, "_priority_level", "LOW")
    setattr(req_a, "_priority_rank", 1)
    setattr(req_b, "_priority_level", "LOW")
    setattr(req_b, "_priority_rank", 1)

    util.global_request_queue.push(0, req_a, prefix_cache_hit_len=0)
    util.global_request_queue.push(0, req_b, prefix_cache_hit_len=0)

    import asyncio

    asyncio.run(util.rebalance_replica_global_waiting_reqs(source_replica_id=0, num_target_migrate_prefill_tokens=1))

    migrated = [req for req in [req_a, req_b] if req._primary_replica == 1]
    assert len(migrated) == 1
    moved = migrated[0]
    assert moved._second_replica == 0
    assert moved._priority_level == "LOW"
    assert moved._priority_rank == 1
    assert hasattr(moved, "_estimated_ttft")


def test_pr2_default_threshold_policy_is_backward_compatible():
    args = _args(enable_shared=True)
    args.priority_policy = "threshold"
    args.dh_first_balance_ttft_thredhold = 5
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=SimpleNamespace(), args=args)

    req_fast = _req(req_id=401, token_count=8)
    req_slow = _req(req_id=402, token_count=8)
    util._set_request_priority(req_fast, predicted_ttft=3.0)
    util._set_request_priority(req_slow, predicted_ttft=8.0)

    assert req_fast._priority_level == "LOW"
    assert req_fast._priority_rank == 1
    assert req_slow._priority_level == "HIGH"
    assert req_slow._priority_rank == 0


def test_pr2_quantile_policy_uses_window_history():
    args = _args(enable_shared=True)
    args.priority_policy = "quantile"
    args.priority_quantile = 0.5
    args.priority_window_size = 3
    args.dh_first_balance_ttft_thredhold = 1
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=SimpleNamespace(), args=args)

    req1 = _req(req_id=411, token_count=8)
    req2 = _req(req_id=412, token_count=8)
    req3 = _req(req_id=413, token_count=8)
    req4 = _req(req_id=414, token_count=8)

    util.update_observed_ttft(2.0)
    util.update_observed_ttft(4.0)
    util.update_observed_ttft(3.0)
    util._set_request_priority(req1, predicted_ttft=2.0)  # quantile([2,3,4])=3 => LOW
    util._set_request_priority(req2, predicted_ttft=4.0)  # quantile([2,3,4])=3 => HIGH
    util._set_request_priority(req3, predicted_ttft=4.1)  # quantile([2,3,4])=3 => HIGH
    util._set_request_priority(req4, predicted_ttft=3.0)  # quantile([2,3,4])=3 => LOW

    assert req1._priority_level == "LOW"
    assert req2._priority_level == "HIGH"
    assert req3._priority_level == "HIGH"
    assert req4._priority_level == "LOW"


def test_pr2_quantile_window_uses_observed_ttft_not_predicted_ttft():
    args = _args(enable_shared=True)
    args.priority_policy = "quantile"
    args.priority_quantile = 0.5
    args.priority_window_size = 4
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=SimpleNamespace(), args=args)

    util.update_observed_ttft(10.0)
    util.update_observed_ttft(20.0)
    req = _req(req_id=421, token_count=8)
    util._set_request_priority(req, predicted_ttft=15.0)  # median([10,20]) = 10 (nearest-rank)
    assert req._priority_level == "HIGH"
    # predicted ttft should not be appended to observed window
    assert list(util._priority_ttft_window) == [10.0, 20.0]


def test_prefix_kv_load_priority_from_request_label():
    req = _req(req_id=431, token_count=8)
    setattr(req, "_priority_level", "HIGH")
    assert get_prefix_kv_load_priority(req) == "high"
    setattr(req, "_priority_level", "LOW")
    assert get_prefix_kv_load_priority(req) == "low"
    delattr(req, "_priority_level")
    assert get_prefix_kv_load_priority(req) == "low"


def test_priority_observed_ttft_callback_on_request_complete():
    args = _args(enable_shared=True)
    state = SharedState(metric_store=_DummyMetricStore(), tokenizer=_DummyTokenizer(), args=args)
    observed = []
    state.set_priority_observed_ttft_callback(lambda ttft: observed.append(ttft))
    request = _req(req_id=441, token_count=8)
    output = SimpleNamespace(success=True, ttft=1.23)

    import asyncio

    asyncio.run(state.on_request_complete(output, request, state.replicas_ip_port[0]))
    assert observed == [1.23]


def test_priority_observed_ttft_callback_not_called_on_failed_request():
    args = _args(enable_shared=True)
    state = SharedState(metric_store=_DummyMetricStore(), tokenizer=_DummyTokenizer(), args=args)
    observed = []
    state.set_priority_observed_ttft_callback(lambda ttft: observed.append(ttft))
    request = _req(req_id=451, token_count=8)
    output = SimpleNamespace(success=False, ttft=9.99)

    import asyncio

    asyncio.run(state.on_request_complete(output, request, state.replicas_ip_port[0]))
    assert observed == []


def test_priority_enabled_false_forces_low_label():
    args = _args(enable_shared=True)
    args.priority_enabled = False
    args.priority_policy = "quantile"
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=SimpleNamespace(), args=args)
    util.update_observed_ttft(1.0)
    req = _req(req_id=461, token_count=8)

    util._set_request_priority(req, predicted_ttft=0.01)
    assert req._priority_level == "LOW"
    assert req._priority_rank == 1


def test_update_observed_ttft_rejects_invalid_values():
    args = _args(enable_shared=True)
    util = DoubleHashGlobalSchedulerUtils(num_replicas=1, shared_state=SimpleNamespace(), args=args)

    util.update_observed_ttft(None)
    util.update_observed_ttft("bad")
    util.update_observed_ttft(-1)
    assert list(util._priority_ttft_window) == []

    util.update_observed_ttft(0.5)
    assert list(util._priority_ttft_window) == [0.5]
