import argparse
import asyncio
from multiprocessing import Process, Event
from dualmap.launcher import SystemLauncher
from types import SimpleNamespace
import os
import sys
import time
import datetime
import signal
import subprocess
from typing import Dict, List, Set, Optional, Tuple
import socket
from dualmap.logger import init_logger

logger = init_logger(__name__)


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed, using stdlib for process management (limited functionality)")


class Config:
    """Unified configuration."""
    START_PROXY_ENABLE = True
    START_SERVICE_ENABLE = False
    STOP_SERVICE_ENABLE = False
    
    # DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
    # VLLM_PORTS = [8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088]
    # VLLM_IP_PORT = "127.0.0.1:8081,127.0.0.1:8082,127.0.0.1:8083,127.0.0.1:8084,127.0.0.1:8085,127.0.0.1:8086,127.0.0.1:8087,127.0.0.1:8088"
    DEVICES = [0,0]
    VLLM_PORTS = [8081,8082]
    VLLM_IP_PORT = "127.0.0.1:8081,127.0.0.1:8082"

    WORKSPACE_PATH = "./evaluation"
    MODEL_PATH = "./models/Qwen/Qwen2.5-7B-Instruct" 
    MODEL = "Qwen/Qwen2.5-7B-Instruct" 
    MODEL_NAME = "Qwen2.5-7B-Instruct" 
    MAX_MODEL_LEN = 20480 # 20480
    WARM_UP_REQUESTS_NUM = 500 # 500  
    WARM_UP_QPS = 0.5 # 0.5 for toolagent, 0.3 for conversation
   
    REPLICIA_DRAM = 64
    KV_CACHE_SIZE_PER_TOKEN = 57344 # 57344 for Qwen2.5-7B-Instruct
    BLOCK_SIZE = 128
    TTFT_SLO = 5 # second
    PREFILL_TPOT = 0.000127# 910b4 and Qwen2.5-7B-Instruct: 0.00016,   910b3 and Qwen2.5-14B-Instruct:0.000127

    PROCESS_CACHE_TTL = 1.0
    VLLM_START_TIMEOUT = 120
    SERVICE_HEALTH_CHECK_TIMEOUT = 20 # seconds
    PROCESS_KILL_TIMEOUT = 10
    PROCESS_CLEANUP_TIMEOUT = 10
    ENABLE_SCALE = False

start_proxy_enable = Config.START_PROXY_ENABLE
start_service_enable = Config.START_SERVICE_ENABLE
stop_service_enable = Config.STOP_SERVICE_ENABLE

server_list: Dict[int, List[int]] = {}

devices = Config.DEVICES
vllm_ports = Config.VLLM_PORTS
replica_num = len(Config.VLLM_PORTS)
model = Config.MODEL
model_name = Config.MODEL_NAME
model_path = Config.MODEL_PATH
max_model_len = Config.MAX_MODEL_LEN
workspace_path = Config.WORKSPACE_PATH
warm_up_requests_num = Config.WARM_UP_REQUESTS_NUM
warm_up_qps = Config.WARM_UP_QPS
tmp_rebalance_thredhold_list = [0.5]
replicas_ip_port = Config.VLLM_IP_PORT
vllm_log_keyword = "YY:DEBUG"
replica_dram = Config.REPLICIA_DRAM
cache_capacity = replica_dram << 30
block_size = Config.BLOCK_SIZE
kv_cache_size_per_token = Config.KV_CACHE_SIZE_PER_TOKEN
ttft_slo = Config.TTFT_SLO
prefill_tpot = Config.PREFILL_TPOT
enable_scale = Config.ENABLE_SCALE

request_active_timeout = 60 # seconds, client will be killed whnen no response after this timeout
requests_num_dataset_start = 0
dh_recompute_punish_ratio = 1
dh_first_balance_ttft_thredhold = int((ttft_slo/2) * (1/prefill_tpot)) 
decode_busy_threshold = ttft_slo # not used
replica_slo_budget = int((ttft_slo/2) * (1/prefill_tpot) * 2)
dh_rebalance_thredhold = int((ttft_slo/2) * (1/prefill_tpot) * 2)
dh_cancel_rebalance_req = False
dh_window_duration = 30
dh_replica_pending_req_threshold = 1
dh_rebalance_waiting_latency_thredhold = 3
preble_window_duration = 3
busy_prefill_interval = 2

def build_namespace(global_scheduler_type: str, qps: float, 
                    model, model_path, model_name, 
                    replica_num, replica_dram,
                    request_num, 
                    request_active_timeout,
                    dataset_type, dataset_file,
                    request_dataset_dir,
                    balance_type, ttft_slo, replica_slo_budget,
                    dh_recompute_punish_ratio, dh_rebalance_thredhold,
                    dh_rebalance_waiting_latency_thredhold,
                    result_path
                    ) -> argparse.Namespace:
    """Build argument namespace."""
    args_dict = {
        "replicas_ip_port": replicas_ip_port,
        "model": model,
        "model_path": model_path,
        "model_name": model_name,
        "replica_num": replica_num,
        "replica_dram": replica_dram,
        "cache_capacity": cache_capacity,
        "block_size": block_size,
        "kv_cache_size_per_token": kv_cache_size_per_token,
        "max_model_len": max_model_len,
        "request_generate_qps": qps,
        "request_num": request_num,
        "warm_up_requests_num": warm_up_requests_num,
        "warm_up_qps": warm_up_qps,
        "requests_num_dataset_start": requests_num_dataset_start,
        "request_active_timeout": request_active_timeout,
        "dataset_type": dataset_type, 
        "dataset_file": dataset_file,
        "request_dataset_dir": request_dataset_dir,
        "global_scheduler_type": global_scheduler_type,
        "balance_type": balance_type, 
        "prefill_tpot": prefill_tpot,
        "decode_busy_threshold": decode_busy_threshold,
        "dh_first_balance_ttft_thredhold":dh_first_balance_ttft_thredhold,
        "dh_rebalance_thredhold": dh_rebalance_thredhold,
        "dh_rebalance_waiting_latency_thredhold":dh_rebalance_waiting_latency_thredhold,
        "dh_recompute_punish_ratio":dh_recompute_punish_ratio,
        "dh_cancel_rebalance_req":dh_cancel_rebalance_req,
        "replica_slo_budget":replica_slo_budget,
        "ttft_slo": ttft_slo,
        "update_replica_info": True,
        "window_duration": 30,
        "dh_window_duration": dh_window_duration,
        "dh_replica_pending_req_threshold":dh_replica_pending_req_threshold,
        "preble_window_duration": preble_window_duration,
        "result_path": result_path,
        "busy_prefill_interval":busy_prefill_interval,
        "enable_scale": enable_scale
    }
    return SimpleNamespace(**args_dict)

def execute_shell_cmd(cmd: str, env: Dict[str, str] = None) -> subprocess.Popen:
    """Run shell command and return process object."""
    return subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        env=env or os.environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

def get_all_child_pids(pid: int) -> Set[int]:
    """Recursively get all child PIDs."""
    child_pids = set()
    if PSUTIL_AVAILABLE:
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child_pids.add(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    else:
        try:
            children_file = f"/proc/{pid}/task/{pid}/children"
            if os.path.exists(children_file):
                with open(children_file, 'r') as f:
                    direct_children = [int(c) for c in f.read().split() if c.strip()]
                    child_pids.update(direct_children)
                    for child_pid in direct_children:
                        child_pids.update(get_all_child_pids(child_pid))
        except (OSError, ValueError, FileNotFoundError):
            pass
    
    return child_pids


def kill_process_tree(pid: int, timeout: int = None) -> bool:
    """
    Kill process tree (including all children).

    Args:
        pid: Process ID
        timeout: Graceful shutdown timeout (seconds), None uses Config.PROCESS_KILL_TIMEOUT
    Returns:
        True if process was killed, False if process does not exist
    """
    if timeout is None:
        timeout = Config.PROCESS_KILL_TIMEOUT
    
    killed_any = False
    
    all_pids = {pid}
    try:
        all_pids.update(get_all_child_pids(pid))
    except Exception as e:
        logger.debug(f"Failed to get child processes {pid}: {e}")

    for target_pid in all_pids:
        if not is_process_alive(target_pid):
            continue
        try:
            os.kill(target_pid, signal.SIGTERM)
            killed_any = True
            logger.debug(f"Sent SIGTERM to process {target_pid}")
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning(f"No permission to kill process {target_pid}, trying process group or sudo")
            try:
                gid = os.getpgid(target_pid)
                os.killpg(gid, signal.SIGTERM)
                killed_any = True
            except (ProcessLookupError, PermissionError):
                try:
                    subprocess.run(
                        f"sudo kill -15 {target_pid}",
                        shell=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL
                    )
                    killed_any = True
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    logger.warning(f"Cannot kill process {target_pid}, may need root")
    
    if killed_any:
        start_time = time.time()
        remaining_pids = {p for p in all_pids if is_process_alive(p)}
        
        while remaining_pids and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            remaining_pids = {p for p in remaining_pids if is_process_alive(p)}
    
    for target_pid in all_pids:
        if not is_process_alive(target_pid):
            continue
            
        try:
            os.kill(target_pid, signal.SIGKILL)
            logger.debug(f"Kill process {target_pid} (SIGKILL)")
            time.sleep(0.2)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                gid = os.getpgid(target_pid)
                os.killpg(gid, signal.SIGKILL)
                logger.debug(f"Kill process group {gid} (SIGKILL)")
                time.sleep(0.2)
            except (ProcessLookupError, PermissionError):
                try:
                    subprocess.run(
                        f"sudo kill -9 {target_pid}",
                        shell=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL
                    )
                except:
                    logger.warning(f"Cannot force kill process {target_pid}")
    
    return killed_any


def is_process_alive(pid: int) -> bool:
    """Check if process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

_process_cache: Dict[str, Set[int]] = {}
_cache_timestamp: Dict[str, float] = {}


def find_processes_by_pattern(pattern: str, cache_ttl: float = None) -> Set[int]:
    """Find PIDs matching command-line pattern (with cache)."""
    if cache_ttl is None:
        cache_ttl = Config.PROCESS_CACHE_TTL
    
    now = time.time()
    if pattern in _process_cache and pattern in _cache_timestamp:
        if now - _cache_timestamp[pattern] < cache_ttl:
            return _process_cache[pattern].copy()
    
    pids = set()
    
    if PSUTIL_AVAILABLE:
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if pattern in cmdline:
                        pids.add(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"psutil find processes failed: {e}")
    else:
        try:
            output = subprocess.check_output(
                f"pgrep -f '{pattern}'",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            for pid_str in output.strip().split():
                if pid_str.strip():
                    pids.add(int(pid_str))
        except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
            pass
    
    _process_cache[pattern] = pids
    _cache_timestamp[pattern] = now
    
    return pids.copy()


def clear_process_cache():
    """Clear process lookup cache."""
    global _process_cache, _cache_timestamp
    _process_cache.clear()
    _cache_timestamp.clear()


def stop_all_vllm_services():
    """Stop all vllm.entrypoints.openai.api_server processes (recursive)."""
    logger.info("=" * 80)
    logger.info("Stopping all VLLM service processes")
    logger.info("=" * 80)
    global server_list
    all_target_pids = set()
    
    for pids in server_list.values():
        all_target_pids.update(pids)
    
    search_patterns = ['vllm.entrypoints.openai.api_server']
    for pattern in search_patterns:
        all_target_pids.update(find_processes_by_pattern(pattern))
    
    all_pids_with_children = set(all_target_pids)
    for pid in all_target_pids:
        try:
            all_pids_with_children.update(get_all_child_pids(pid))
        except Exception:
            pass
    
    valid_pids = {pid for pid in all_pids_with_children if is_process_alive(pid)}
    
    if not valid_pids:
        logger.info("No processes to kill")
        server_list.clear()
        clear_process_cache()
        return
    
    logger.info(f"Killing {len(valid_pids)} processes: {sorted(valid_pids)}")
    sorted_pids = sorted(valid_pids, reverse=True)
    logger.info("Step 1: Sending SIGTERM...")
    for pid in sorted_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.debug(f"  Sent SIGTERM to {pid}")
        except (ProcessLookupError, PermissionError):
            pass
    
    logger.info(f"Step 2: Waiting for exit (up to {Config.PROCESS_CLEANUP_TIMEOUT}s)...")
    start_time = time.time()
    remaining_pids = {p for p in sorted_pids if is_process_alive(p)}
    
    while remaining_pids and (time.time() - start_time) < Config.PROCESS_CLEANUP_TIMEOUT:
        time.sleep(0.5)
        remaining_pids = {p for p in remaining_pids if is_process_alive(p)}
        if remaining_pids and len(remaining_pids) % 5 == 0:
            logger.debug(f"  Still {len(remaining_pids)} running...")
    if remaining_pids:
        logger.info(f"Step 3: Force killing {len(remaining_pids)} (SIGKILL)...")
        for pid in remaining_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                logger.debug(f"  Killed {pid}")
                time.sleep(0.1)
            except (ProcessLookupError, PermissionError):
                try:
                    gid = os.getpgid(pid)
                    os.killpg(gid, signal.SIGKILL)
                    logger.debug(f"  Killed process group {gid}")
                    time.sleep(0.1)
                except (ProcessLookupError, PermissionError, OSError):
                    logger.warning(f"  Cannot kill {pid}, may need root")
    clear_process_cache()
    logger.info("Step 4: Verifying...")
    final_check_patterns = [ 'vllm.entrypoints.openai.api_server']
    remaining_after_kill = set()
    
    for pattern in final_check_patterns:
        remaining_after_kill.update(find_processes_by_pattern(pattern))
    
    if remaining_after_kill:
        logger.warning(f"  {len(remaining_after_kill)} still running: {sorted(remaining_after_kill)}")
        for pid in remaining_after_kill:
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.1)
            except:
                pass
    else:
        logger.info("All processes stopped")
    server_list.clear()
    clear_process_cache()
    logger.info("=" * 80)
    logger.info("Process cleanup done")
    logger.info("=" * 80)

async def start_single_vllm_service_async(vllm_port: int,  device: int, sys_args) -> Tuple[int, List[int], bool]:
    """Async vllm service starter. Returns: (port, PID list, success)."""
    log_prefix = f"[device {device}, port {vllm_port}]"
    os.makedirs(sys_args.result_path, exist_ok=True)
    log_file = f"{sys_args.result_path}/cache_hit_ratio_{vllm_port}.log"
    
    # KV transfer config
    
    cmd = (
        f"export CUDA_VISIBLE_DEVICES={device} && "
        f"python -u -m vllm.entrypoints.openai.api_server "
        f"--model {Config.MODEL_PATH} "
        f"--max-num-seqs=256 "
        f"--max-model-len={Config.MAX_MODEL_LEN} "
        f"--max-num-batched-tokens={Config.MAX_MODEL_LEN} "
        f"--dtype=float16 "
        f"--tensor-parallel-size=1 "
        f"--block-size=128 "
        f"--host=0.0.0.0 "
        f"--port={vllm_port} "
        f"--gpu-memory-utilization=0.9 "
        f"--trust-remote-code "
        f"--served-model-name {Config.MODEL_NAME} "
        #f"--enforce-eager | "
        #f"grep -i {vllm_log_keyword} > {log_file}"
        f"> {log_file}"
    )
    print(f"start server:{cmd}") 
    logger.info(f"{log_prefix} Starting vllm.entrypoints.openai.api_server")
    logger.debug(f"{log_prefix} cmd: {cmd}")
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        start_time = time.time()
        check_interval = 10
        port_ready = False
        
        while time.time() - start_time < Config.VLLM_START_TIMEOUT:
            try:
                curl_cmd = f"curl -X GET -I http://localhost:{vllm_port}/health -m 1 -s"
                curl_process = await asyncio.create_subprocess_shell(
                    curl_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    curl_process.communicate(),
                    timeout=2.0
                )
                stdout_str = stdout.decode()
                if "200" in stdout_str or "HTTP/1.0 200 OK" in stdout_str:
                    port_ready = True
                    logger.debug(f"{log_prefix} Health check OK: {stdout_str.strip()}")
                    break
                else:
                    logger.debug(f"{log_prefix} Health check failed: {stdout_str.strip()}")
            except asyncio.TimeoutError:
                logger.debug(f"{log_prefix} Health check timeout, retry in {check_interval}s")
            except Exception as e:
                logger.debug(f"{log_prefix} Health check error: {str(e)}")
            
            await asyncio.sleep(check_interval)
        
        if not port_ready:
            logger.warning(f"{log_prefix} vllm health check timeout, continuing to get PIDs")
        pids = []
        try:
            pgrep_process = await asyncio.create_subprocess_shell(
                f"pgrep -f 'vllm.entrypoints.openai.api_server.*{vllm_port}'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await pgrep_process.communicate()
            if stdout:
                pids = [int(pid_str.strip()) for pid_str in stdout.decode().split() if pid_str.strip()]
                logger.debug(f"{log_prefix} Got PIDs: {pids}")
        except Exception as e:
            logger.warning(f"{log_prefix} Failed to get vllm PIDs: {str(e)}")
            if process.pid:
                main_pid = process.pid
                child_pids = get_all_child_pids(main_pid)
                pids = [main_pid] + list(child_pids)
                logger.debug(f"{log_prefix} Fallback PIDs: {pids}")
        if pids:
            logger.info(f"{log_prefix} vllm started, PID: {pids}")
            return vllm_port, pids, True
        else:
            logger.error(f"{log_prefix} vllm no PIDs")
            if process.pid:
                kill_process_tree(process.pid)
            return vllm_port, [], False
            
    except Exception as e:
        logger.error(f"{log_prefix} vllm start failed: {e}")
        return vllm_port, [], False


async def start_all_vllm_services_async(sys_args) -> bool:
    """Start all vllm services sequentially. Returns: all succeeded."""
    global server_list
    server_list.clear()
    failed_ports = []
    for device, vllm_port in zip(Config.DEVICES, Config.VLLM_PORTS):
        logger.info(f"Starting vllm (port: {vllm_port}, device: {device})")
        try:
            result = await start_single_vllm_service_async(vllm_port, device, sys_args)
            vllm_port_res, pids, success = result
            if success and pids:
                server_list[vllm_port_res] = pids
                logger.info(f"vllm (port: {vllm_port_res}) started")
            else:
                failed_ports.append(vllm_port_res)
                logger.error(f"vllm (port: {vllm_port_res}) failed")
        except Exception as e:
            logger.error(f"vllm (port: {vllm_port}) start error: {e}")
            failed_ports.append(vllm_port)
    if failed_ports:
        logger.error(f"vllm start failed, ports: {failed_ports}")
        return False
    logger.info(f"All {len(server_list)} vllm services started")
    return True


def _start_all_vllm_services_helper(sys_args):
    """Start all vllm.entrypoints.openai.api_server services."""
    global server_list
    server_list.clear()
    if not (len(Config.DEVICES) == len(Config.VLLM_PORTS)):
        logger.error(f"Array length mismatch: devices={len(Config.DEVICES)}, vllm_ports={len(Config.VLLM_PORTS)}")
        return False
    logger.info(f"Starting {len(Config.VLLM_PORTS)} vllm services...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vllm_start_success = loop.run_until_complete(start_all_vllm_services_async(sys_args))
        loop.close()
        
        if not vllm_start_success:
            logger.error("vllm start failed")
            return False
    except Exception as e:
        logger.error(f"vllm async start error: {e}")
        return False

    return True



def vllm_services_health_check(result_path, sync_log_enable) -> bool:
    """Health check for vllm.entrypoints.openai.api_server."""
    os.makedirs(result_path, exist_ok=True)
    log_file = f"{result_path}/services_health.log"
    all_valid = True
    
    global server_list
    
    active_vllm_pids = set()
    try:
        output = subprocess.check_output(
            "pgrep -f 'vllm.entrypoints.openai.api_server'",
            shell=True,
            text=True,
            timeout=5
        )
        active_vllm_pids = set(int(p) for p in output.strip().split())
    except:
        pass
    
    for vllm_port, pids in server_list.items():
        valid_pids = [pid for pid in pids if pid in active_vllm_pids]
        server_list[vllm_port] = valid_pids
        
        if len(valid_pids) < 1:
            logger.warning(f"vllm (port {vllm_port}) too few processes (current: {len(valid_pids)})")
            all_valid = False

    return all_valid


def start_all_vllm_services(sys_args):
    """Service manager: start vllm.entrypoints.openai.api_server."""
    max_retries = 3
    for attempt in range(max_retries):
        logger.info(f"\n=== Start attempt {attempt+1}/{max_retries} ===")
        stop_all_vllm_services()
        time.sleep(5)
        if not _start_all_vllm_services_helper(sys_args):
            logger.warning("Service start failed, retrying...")
            continue
        logger.info("start_all_services, process list:")
        logger.info(f"  vllm: {server_list}")
        time.sleep(Config.SERVICE_HEALTH_CHECK_TIMEOUT)
        if vllm_services_health_check(sys_args.result_path, False):
            logger.info("All services started")
            logger.info("health_check, process list:")
            logger.info(f"  vllm: {server_list}")
            return True
        else:
            logger.warning("Health check failed, retrying...")
    logger.error("Max retries exceeded, start failed")
    return False


async def _async_worker(launcher, result_path):
    """Async worker."""
    try:
        await launcher.run()
        while await launcher.is_request_active():
            if not vllm_services_health_check(result_path, True):
                print("All services started")
                print("health_check, process list:", server_list)
                break
            await asyncio.sleep(60)
        logger.debug("launcher.is_request_active()==false")
    finally:
        logger.debug("client stopped")


def single_experiment(global_scheduler_type: str, qps: float, sys_args):
    """Single experiment process."""
    logger.debug(f"\n=== Start experiment {sys_args.global_scheduler_type} {sys_args.balance_type}/{sys_args.request_generate_qps} ===")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        launcher = SystemLauncher(sys_args)
        loop.run_until_complete(launcher.initialize(sys_args))
        loop.run_until_complete(_async_worker(launcher, sys_args.result_path))
    finally:
        loop.close()
        logger.debug(f"=== End experiment {sys_args.global_scheduler_type} {sys_args.balance_type}/{sys_args.request_generate_qps} ===")


def start_exp(global_scheduler_type_list,balance_type_list,qps_list,request_num,dataset_type, dataset_file):
    """Main control flow."""
    for tmp_rebalance_thredhold in tmp_rebalance_thredhold_list:
        dh_rebalance_thredhold = int(ttft_slo * (1/prefill_tpot) * tmp_rebalance_thredhold)
        dh_first_balance_ttft_thredhold = dh_rebalance_thredhold
        replica_slo_budget = dh_rebalance_thredhold
        for qps_str in (qps_list.split(",") if isinstance(qps_list, str) else qps_list):
            qps = round(float(qps_str.strip()) if isinstance(qps_str, str) else float(qps_str), 1)
            for global_scheduler_type in global_scheduler_type_list:
                for balance_type in balance_type_list:
                    global_scheduler_dir = ""
                    if global_scheduler_type != "dualmap":
                        if balance_type == "": 
                            global_scheduler_dir = global_scheduler_type
                        else:
                            continue
                    elif global_scheduler_type == "dualmap":
                        if balance_type == "":
                            continue
                        else:
                            global_scheduler_dir = balance_type
                    result_path = ""
                    result_path = os.path.join(
                        workspace_path,
                        'result',
                        'data',
                        model_name,
                        dataset_type,
                        f'req{request_num}-warm{warm_up_requests_num}-cache{replica_dram}G',
                        f'replica{replica_num}-qps{qps}',
                        global_scheduler_dir
                    )


                    request_dataset_dir = os.path.join(
                        workspace_path,
                        'dataset',
                        'mooncake'
                    )                        

                    sys_args = build_namespace(global_scheduler_type, qps, 
                            model, model_path, model_name, 
                            replica_num, replica_dram,
                            request_num, 
                            request_active_timeout,
                            dataset_type, dataset_file,
                            request_dataset_dir,
                            balance_type, ttft_slo, replica_slo_budget,
                            dh_recompute_punish_ratio,dh_rebalance_thredhold,
                            dh_rebalance_waiting_latency_thredhold,
                            result_path
                            ) 
                    if stop_service_enable:
                        logger.debug(f"stop_vllm_services")
                        stop_all_vllm_services()
                        return

                    if start_service_enable:
                        if not start_all_vllm_services(sys_args):
                            stop_all_vllm_services()
                            time.sleep(5)
                            logger.debug(f"vllm server start failed, skip experiment {global_scheduler_dir}/{qps}")
                            continue
                    if start_proxy_enable:
                        p = Process(target=single_experiment, args=(global_scheduler_type, qps, sys_args))
                        p.start()
                        p.join()
                        logger.debug(f"=== Done experiment {global_scheduler_dir}/{qps} ===")
                        time.sleep(5)
                        logger.debug(f"stop_vllm_services, {global_scheduler_dir}/{qps}")


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM and cleanup processes."""
    signal_name = signal.Signals(signum).name
    logger.warning(f"Received {signal_name} ({signum}), cleaning up...")
    try:
        stop_all_vllm_services()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    logger.info("Exiting")
    sys.exit(0)


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Process dataset for DualMap experiments")
    parser.add_argument('--replicas_ip_port', type=str, default="127.0.0.1:8081,127.0.0.1:8082,127.0.0.1:8083,127.0.0.1:8084,127.0.0.1:8085,127.0.0.1:8086,127.0.0.1:8087,127.0.0.1:8088", help='replicas ip port')  
    parser.add_argument('--model_path', type=str, default="./models/Qwen/Qwen2.5-7B-Instruct", help='model path')  
    parser.add_argument('--model_name', type=str, default="Qwen2.5-7B-Instruct", help='model name')  
    parser.add_argument('--max_model_len', type=int, default=20480, help='max model len') 
    parser.add_argument('--block_size', type=int, default=128, help='block size')  
    parser.add_argument('--kv_cache_size_per_token', type=int, default=57344, help='kv cache size per token') 
    parser.add_argument('--replica_dram_size', type=int, default=64, help='replica dram size')
    parser.add_argument('--qps', type=str, default="3", help='qps') 
    parser.add_argument('--prefill_tpot', type=float, default=0.000127, help='prefill tpot')  
    parser.add_argument('--global_scheduler_type', type=str, default="cache_affinity", help='comma-separated scheduler types, e.g. "cache_affinity,dualmap"')  
    parser.add_argument('--ttft_slo', type=int, default=5, help='ttft_slo seconds')  
    parser.add_argument('--request_num', type=int, default=8000, help='request num')  
    parser.add_argument('--dataset_file', type=str, default="./evaluation/dataset/mooncake/processed_toolagent_trace.jsonl", help='dataset file')
    parser.add_argument('--dataset_type', type=str, default="toolagent", help='origin dataset type')

    args = parser.parse_args()

    replicas_ip_port = args.replicas_ip_port
    model_path = args.model_path
    model_name = args.model_name
    model = getattr(args, "model", None) or args.model_path
    max_model_len = args.max_model_len
    block_size = args.block_size
    kv_cache_size_per_token = args.kv_cache_size_per_token
    replica_dram = args.replica_dram_size
    cache_capacity = replica_dram << 30
    replica_num = len(args.replicas_ip_port.split(","))
    prefill_tpot = args.prefill_tpot
    ttft_slo = args.ttft_slo

    
    global_scheduler_type = args.global_scheduler_type
    qps = args.qps
    request_num = args.request_num
    dataset_file = args.dataset_file
    dataset_type = args.dataset_type
    

    # balance_type_list: dualmap variants for ablation‘2026 ablation (ICLR 2026 figure 5)
    balance_type_list = ["", "dualmap"] 
    global_scheduler_type_list = [global_scheduler_type]
    qps_list = [qps]
    start_exp(global_scheduler_type_list,balance_type_list,qps_list,request_num,dataset_type,dataset_file)
