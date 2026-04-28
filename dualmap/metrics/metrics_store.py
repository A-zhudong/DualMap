import os
import aiofiles
import csv
import asyncio
from collections import defaultdict
import io
from dualmap.logger import init_logger

logger = init_logger(__name__)

class RooundInfo():

    def __init__(self):
        self.session_num = 0
        self.request_num = 0
        self.request_latency_list = []
        self.ttft_list = []
        self.decode_token_time_list = []
        self.output_decode_token_num_list = []       

    def save_info(self,
                  request_latency: float,
                  time_to_first_token: float,
                  decode_token_time: float,
                  output_token_len: int):
        self.request_num += 1
        self.request_latency_list.append(request_latency)
        self.ttft_list.append(time_to_first_token)
        self.decode_token_time_list.append(decode_token_time)
        self.output_decode_token_num_list.append(output_token_len)

class MetricsStore:
    def __init__(self, args):
        self.csv_dir = args.result_path
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_filename = f"{self.csv_dir}/request_metrics.csv"
        self.lock = asyncio.Lock()
        self.data_cache = defaultdict(dict)
        self.dirty = False 

    async def load_cache(self):
        try:
            async with aiofiles.open(self.csv_filename, mode="r", encoding="utf-8") as csvfile:
                content = await csvfile.read()
                lines = content.splitlines()
                reader = csv.DictReader(lines)
                for row in reader:
                    self.data_cache[row['request_id']] = row
        except FileNotFoundError:
            pass

    async def save_cache(self):
        if self.dirty:
            try:
                output = io.StringIO()
                fieldnames = ['request_id', 'dataset_type','request_start_time', 'request_end_time', 'native_session_id', 'round_id', 'replica_id', 
                            'time_to_first_token', 'request_latency', 'TPS(tokens/s)', 'tpot(ms)', 'num_request_pending', 'input_len', 
                            'output_len', 'pd_ratio', 'actual_num_prefill_tokens', 'req_arrived_at','time_interval','rounting_cache_hit_max',
                            'is_dh_cache_affinity','is_dh_least_loaded','is_dh_cache_affinity_least_loaded',
                            'priority_enabled','priority_policy','priority_level','priority_predicted_ttft',
                            'priority_threshold','priority_quantile','priority_window_size']
                writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
                writer.writeheader()
                for row in self.data_cache.values():
                    writer.writerow(row)
                csv_content = output.getvalue()
                output.close()

                async with aiofiles.open(self.csv_filename, mode="w", encoding="utf-8") as csvfile:
                    await csvfile.write(csv_content)
                self.dirty = False
            except Exception as e:
                logger.error(f"Failed to save cache to CSV: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())

    async def insert_metrics(self, data):
        async with self.lock:
            self.data_cache[data['request_id']] = data
            self.dirty = True

    async def update_metrics(self, request_id, updates):
        async with self.lock:
            if request_id in self.data_cache:
                self.data_cache[request_id].update(updates)
                logger.debug(f"update_metrics:request_id={request_id}")
                self.dirty = True

    async def sync_cache(self):
        await self.save_cache()
