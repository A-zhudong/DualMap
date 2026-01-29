from transformers import AutoTokenizer
from dualmap.scheduler.utils.shared import SharedState
from dualmap.metrics.metrics_store import MetricsStore
from dualmap.scheduler.global_scheduler.request_router_proxy import RequestRouterProxy
from dualmap.scheduler.global_scheduler.preble_global_scheduler import PrebleGlobalScheduler
from dualmap.scheduler.global_scheduler.cache_affinity_global_scheduler import CacheAffinityGlobalScheduler
from dualmap.request_generator.request_generator import RequestGenerator
from dualmap.scheduler.global_scheduler.round_robin_global_scheduler import RoundRobinGlobalScheduler
from dualmap.scheduler.global_scheduler.min_pending_input_len_global_scheduler import MinPendingInputGlobalScheduler
from dualmap.scheduler.global_scheduler.double_hash_global_scheduler import DoubleHashGlobalScheduler
from dualmap.scheduler.global_scheduler.min_ttft_scheduler import MinTTFTGlobalScheduler
from dualmap.logger import init_logger

logger = init_logger(__name__)

class SystemLauncher:
    def __init__(self, args):
        self._num_replicas = len(args.replicas_ip_port.split(','))
        self._global_scheduler_type = args.global_scheduler_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.metric_store = MetricsStore(args)
        self.shared_state = SharedState(self.metric_store, self.tokenizer, args)
        self._request_router_proxy = RequestRouterProxy(self.shared_state, self._global_scheduler_type, args.balance_type, args)
        self._request_generator = RequestGenerator(self.shared_state, self.tokenizer, self._num_replicas, args)

    async def is_request_active(self) -> bool:
        if self._request_generator is None:
            return True
        active = self._request_generator.is_request_active()
        if not active:
            await self.metric_store.sync_cache()
        return active

    async def initialize(self, args):
        self._init_router(args)
        
    def _init_router(self, args):
        logger.info(f"init_router: {self._global_scheduler_type}")
        if self._global_scheduler_type == "round_robin":
            self._request_router_proxy.global_scheduler = RoundRobinGlobalScheduler(
                num_replicas=self._num_replicas,
                shared_state = self.shared_state,
                args = args
            )
        elif self._global_scheduler_type == "min_pending_input":
            self._request_router_proxy.global_scheduler = MinPendingInputGlobalScheduler(
                num_replicas=self._num_replicas,
                shared_state = self.shared_state,
                args = args
            )
        elif self._global_scheduler_type == "preble":
            self._request_router_proxy.global_scheduler = PrebleGlobalScheduler(
                num_replicas=self._num_replicas,
                window_duration = args.preble_window_duration,
                shared_state = self.shared_state,
                args = args
            )
        elif self._global_scheduler_type == "cache_affinity":
            self._request_router_proxy.global_scheduler = CacheAffinityGlobalScheduler(
                num_replicas=self._num_replicas,
                window_duration = args.window_duration,
                shared_state = self.shared_state,
                args = args
            )
        elif self._global_scheduler_type == "dualmap":
            self._request_router_proxy.global_scheduler = DoubleHashGlobalScheduler(
                num_replicas=self._num_replicas,
                window_duration = args.dh_window_duration,
                balance_type = args.balance_type,
                shared_state = self.shared_state,
                args = args
            )
        elif self._global_scheduler_type == "min_ttft":
            self._request_router_proxy.global_scheduler = MinTTFTGlobalScheduler(
                num_replicas=self._num_replicas,
                shared_state = self.shared_state,
                args = args
            )

    async def start_client(self):
        await self._request_generator.generate_from_file()
    
    async def start_global_scheduler(self):
        await self._request_router_proxy.start()

    async def run(self):
        await self._request_router_proxy.start()
        await self._request_generator.generate_from_file()

    async def stop(self):
        await self._request_router_proxy.stop()