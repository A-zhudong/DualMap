import requests
import aiohttp
import asyncio
import time
import json
import sys
import traceback
from typing import List, Optional
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.entities.request import Request
from dualmap.entities.replica import Replica
from dualmap.logger import init_logger
logger = init_logger(__name__)
last_request_time = Optional[float]


def get_prefix_kv_load_priority(request: Request) -> str:
    priority_level = getattr(request, "_priority_level", "LOW")
    return "high" if str(priority_level).upper() == "HIGH" else "low"

def update_request_time():
    # logger.debug(f'update_request_time')
    last_request_time = time.perf_counter() 

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

async def async_send_request(metric_store, result_path, model_name, 
        replica_id: int = None, native_session_id: str = None, 
        target_ip_port=None, request: Request = None,
        replica: Replica = None):
    if request is None or target_ip_port is None or replica is None:
        logger.error(f"async_send_request:invalid parameters:request={request},target_ip_port={target_ip_port},replica={replica}")
        return None  

    output_token_len = 0
    request_start_time = min(time.perf_counter(),request._arrived_at)
    most_recent_timestamp = request_start_time
    time_to_first_token = 0.0 # ttft
    first_token_flag = False
    interList: List[float] = []
    generator_text = ""
    request_latency = 0.0
    num_request_pending = -1
    # for preble
    start_time = time.time()
    ttft = 0
    output = RequestFuncOutput()
    scheduling_overhead = time.time() - start_time
    output.request_id = str(request._id)
    output.session_id = str(request._session_id)
    output.round_id = str(request._round_id)
    output.prompt_text = request._prompts
    output.prompt_len = len(request._prompts)
    output.runtime_selected = replica_id
    num_request_pending = len(replica.pending_requests)

    replica_url = f"http://{target_ip_port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": request._prompts}],
        "session_id": request._session_id,
        "over_flow": False,
        "n": 1,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": request._output_len,
        "stream": True
    }
    payload.setdefault("kv_transfer_params", {})
    payload["kv_transfer_params"]["prefix_kv_load_priority"] = get_prefix_kv_load_priority(request)
    headers = {
        'Content-Type': 'application/json'
    }
    logger.info(f"async_send_request:request._id={request._id},session={native_session_id},{target_ip_port}")

    data = {
        "request_id": str(request._id),
        "dataset_type": request._dataset_type,
        "request_start_time": request_start_time,
        "request_end_time": 0,
        "native_session_id": native_session_id,
        "round_id": request._round_id,
        "replica_id": replica_id,
        "time_to_first_token": 3600000,
        "request_latency": 3600000,
        "TPS(tokens/s)": 3600000,
        "tpot(ms)": 3600000,
        "num_request_pending": num_request_pending,
        "input_len": request._num_prefill_tokens,
        "output_len": request._output_len,
        "pd_ratio": round(request._num_prefill_tokens/request._output_len,2),
        "actual_num_prefill_tokens": request._actual_num_prefill_tokens,
        "request_end_time": time.perf_counter(),
        "req_arrived_at": request._arrived_at,
        "time_interval": request._time_interval,
        "rounting_cache_hit_max": request._rounting_cache_hit_max,
        "is_dh_cache_affinity": request._is_dh_cache_affinity,
        "is_dh_least_loaded": request._is_dh_least_loaded,
        "is_dh_cache_affinity_least_loaded":request._is_dh_cache_affinity_least_loaded,
        "priority_enabled": getattr(request, "_priority_enabled", 0),
        "priority_policy": getattr(request, "_priority_policy", ""),
        "priority_level": getattr(request, "_priority_level", "LOW"),
        "priority_predicted_ttft": getattr(request, "_estimated_ttft", -1),
        "priority_threshold": getattr(request, "_priority_threshold", -1),
        "priority_quantile": getattr(request, "_priority_quantile", -1),
        "priority_window_size": getattr(request, "_priority_window_size", -1),
    }
    await metric_store.insert_metrics(data)

    REQUEST_TIMEOUT_SECONDS = 3600 * 100
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    response_status = 0
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url=replica_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_status = 200
                        await replica.add_request(request)
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ").strip()
                            if chunk == "[DONE]":
                                request_latency = time.perf_counter() - request_start_time
                                output.success = True
                                output.request_latency = time.perf_counter() - request_start_time # for preble
                                replica.complete_request_decode(request, generator_text)
                            else:
                                data = json.loads(chunk)
                                timestamp = time.perf_counter()
                                delta = data["choices"][0]["delta"]
                                
                                if 'usage' in data:
                                    output_token_len = data["usage"]["completion_tokens"]
                                    output.output_len = output_token_len
                                    output.max_new_tokens = output_token_len
                                
                                if delta.get("content", None): 
                                    if first_token_flag is False: # First token
                                        logger.debug(f'async_send_request: request finish prefill, {request._id}')
                                        time_to_first_token = time.perf_counter() - request_start_time
                                        ttft = time_to_first_token # for preble
                                        output.ttft = ttft # for prebel
                                        first_token_flag = True
                                        logger.info(f"async_send_request: prefill completed: req={request._id},replica_id={replica_id}")
                                        success = await replica.complete_request_prefill(replica_id, request, ttft)
                                        if not success:
                                            logger.warning(f"Request {request} not found in replica {replica_id}")
                                    else: #decode
                                        interList.append(timestamp - most_recent_timestamp)
                                        output.itl.append(timestamp - most_recent_timestamp) # for prebel
                                    generator_text += delta["content"]  # total out_put text
                                most_recent_timestamp = timestamp  
                    else:
                        output.error = response.reason or ""
                        output.success = False # for preble
                        logger.debug(f"response.status={response.status}:request._id={request._id},session={native_session_id},{target_ip_port}")
                        logger.debug(f"bad prompts:{target_ip_port}:output.error ={output.error}:request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                logger.debug(f'async_send_request:Exception: {target_ip_port}: {output.error}')

    except asyncio.CancelledError as e:
        logger.debug(f"{target_ip_port}:Request {request._id} was cancelled:{str(e)}")
        output.success = False
        output.error = "Request cancelled"
    except Exception as e:
        logger.debug(f"{target_ip_port}:Request {request._id} failed: {str(e)}")
        output.success = False
        if not output.error:
            output.error = str(e)

    update_request_time()
    if request._id % 10 == 0:
        await metric_store.save_cache()
    # preble: throughput as token generated per second
    output.scheduling_overhead = scheduling_overhead
    if output.success:
        # logger.debug(f"response.status={200}:request._id={request._id},session={native_session_id},{target_ip_port}")
        logger.debug(f"good request:{target_ip_port}:good,request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
        output.tpot = (output.request_latency - output.ttft) / max(1, request._output_len)
        updates = {
            "time_to_first_token": ttft,
            "request_end_time": time.perf_counter(),
            "request_latency": round(request_latency, 4),
            "TPS(tokens/s)": round(request._output_len/request_latency, 4),
            "tpot(ms)": round(output.tpot*1000, 4),
            "actual_num_prefill_tokens": request._actual_num_prefill_tokens,
            "priority_enabled": getattr(request, "_priority_enabled", 0),
            "priority_policy": getattr(request, "_priority_policy", ""),
            "priority_level": getattr(request, "_priority_level", "LOW"),
            "priority_predicted_ttft": getattr(request, "_estimated_ttft", -1),
            "priority_threshold": getattr(request, "_priority_threshold", -1),
            "priority_quantile": getattr(request, "_priority_quantile", -1),
            "priority_window_size": getattr(request, "_priority_window_size", -1),
        }
        await metric_store.update_metrics(str(request._id), updates) 
    else:
        logger.debug(f"failed:request._id={request._id},session={native_session_id},{target_ip_port}")
        logger.debug(f"bad request:{target_ip_port}:good,request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
        logger.debug(f"output failed:request._id={request._id},session={native_session_id},{target_ip_port}")
        await replica.abort_request(request)
    return output
