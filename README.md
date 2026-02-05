# DualMap

DualMap is a dual-mapping scheduling strategy designed for distributed large language model (LLM) serving, aiming to achieve both cache affinity and load balancing. By employing a dual-hash mapping mechanism based on request prompts, DualMap generates two candidate instances for each request and intelligently selects the optimal one according to the current system state. This design enhances the co-location probability of requests sharing the same prefix while achieving global load balance through the **power of two choices.**  Building on this foundation, DualMap further incorporates SLO-aware routing, hotspot-aware rebalancing to ensure efficient cache reuse and balanced load distribution under dynamic and skewed workloads. These designs significantly enhance the effective request capacity of system and reduce the per-request inference cost.

## Prerequisites

- **Hardware**: ≥ 8 GPUs (for full-scale experiments), DRAM ≥ 512GB
- **Software**: Python 3.10+, PyTorch2.3.0+, vLLM >=0.4.2, HuggingFace `transformers`

## Installation

```bash
conda create -n dualmap python=3.10 -y
conda activate dualmap
cd DualMap
pip install -r ./requirements.txt
```



## Getting Start

### 1. Dataset Preprocessing

Process the [Mooncake](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces) open-source dataset:

```bash
# Qwen2.5-7B-Instruct for mooncake date set
# toolagent
python evaluation/run/process_dataset.py --model_path ./models/Qwen/Qwen2.5-7B-Instruct --origin_dataset_file ./evaluation/dataset/toolagent_trace.jsonl --processed_dataset_file ./evaluation/dataset/mooncake/processed_toolagent_trace.jsonl
# conversation 
python evaluation/run/process_dataset.py --model_path ./models/Qwen/Qwen2.5-7B-Instruct --origin_dataset_file ./evaluation/dataset/conversation_trace.jsonl --processed_dataset_file ./evaluation/dataset/mooncake/processed_conversation_trace.jsonl
```

replace  `./models/Qwen/Qwen2.5-7B-Instruct` with your own model path

 Processed data saved to `./evaluation/dataset/mooncake/processed_toolagent_trace.jsonl` and `./evaluation/dataset/mooncake/processed_conversation_trace.jsonl`

### 2. Launch Inference Instances (via vLLM)

vllm>=0.4.2

Start replicas of the Qwen/Qwen2.5-7B-Instruct model (example for 8 ports):

```bash
# Start replica 1 (port 8081)
python -u -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen/Qwen2.5-7B-Instruct \  # local model path
    --max-num-seqs=256 \
	--max-model-len=20480 \
    --max-num-batched-tokens=20480 \
	--dtype=float16 \
    --tensor-parallel-size=1 \
	--block-size=128 \
    --host=0.0.0.0 \
    --port=8081 \
    --gpu-memory-utilization=0.9 \
    --trust-remote-code \
    --served-model-name "Qwen2.5-7B-Instruct"
```

Repeat the above command for ports **8082–8088** (each bound to a different GPU) to launch the remaining inference instances.

> **Note:** In our experiments, we extend **vLLM** with **DRAM-based KV-cache management**. **LMCache** or  **Mooncake** can serve as a *drop-in replacement* for this DRAM-based KV-cache management module. Each inference instance is allocated 64 GB of DRAM for KV-cache management.

### 3. Run Experiments

```python
python ./start_up.py \
--replicas_ip_port "127.0.0.1:8081,127.0.0.1:8082,127.0.0.1:8083,127.0.0.1:8084,127.0.0.1:8085,127.0.0.1:8086,127.0.0.1:8087,127.0.0.1:8088" \
--model_path "./models/Qwen/Qwen2.5-7B-Instruct" \
--model_name "Qwen2.5-7B-Instruct" \
--max_model_len 20480 \
--block_size 128 \
--kv_cache_size_per_token 57344 \
--replica_dram_size 64 \
--qps 10 \
--prefill_tpot 0.00016 \
--global_scheduler_type "cache_affinity" \
--ttft_slo 5 \
--request_num 8000 \
--dataset_file "./evaluation/dataset/mooncake/processed_toolagent_trace.jsonl" \
--dataset_type "toolagent"
```



#### Parameter Descriptions

| Parameter                   | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| **replicas_ip_port**        | IP and port list of inference instances launched in step 2, separated by commas. |
| **model_path**              | The local model path, consistent with `model` in step 2.     |
| **model_name**              | Same as `served-model-name` in step 2.                       |
| **max_model_len**           | Maximum model length, consistent with `--max-model-len` in step 2. |
| **block_size**              | Same as `--block-size` in step 2.                            |
| **kv_cache_size_per_token** | The KV-cache size per token, determined by model and dtype. For example, `Qwen2.5-7B-Instruct` with `float16` → 57344. |
| **replica_dram_size**       | The amount of DRAM (in GB) allocated per replica for KV-cache management. |
| **qps**                     | Query-per-second (QPS) rates at which the client sends requests to the global scheduler. |
| **prefill_tpot**            | The average processing time per token (in seconds) during the prefill phase on a single inference instance under interference-free conditions. |
| **global_scheduler_type**   | Types of global schedulers to be compared (e.g., `"cache_affinity, min_pending_input, min_ttft, preble,dualmap"`). |
| **ttft_slo**                | Seconds                                                      |
| **request_num**             | Number of requests which the client sends to the global scheduler. |
| **dataset_type**            | Type of dataset: `toolagent` or `conversation`.              |
| **dataset_file**            | Dataset file: processed data file in `step 1. Dataset Preprocessing` |

Experiment outputs will be saved in the directory:`./evaluation/result/data/*`

> Note: Restart all inference instances for another experiment set to ensure environmental consistency.



## Evaluations

### Process Results

```python
python evaluation/plots/dataset_performance/goodput_and_effective_req_capacity.py \
--model_name "Qwen2.5-7B-Instruct" \
--replica_dram_size 64 \
--qps_list "1,2,3,4,5" \
--dataset_type "toolagent" \
--global_scheduler_type_list "cache_affinity,min_pending_input,min_ttft,preble,dualmap" \
--replica_num 8 \
--request_num 8000 \
--ttft_slo 5
```

Processed experiment data and pics will be saved in the directory:`./evaluation/result/processed_data/*`

#### Parameter Descriptions

`model_name,replica_dram_size,qps_list,dataset_type,global_scheduler_type_list,request_num,ttft_slo` are same as  `3. Run Experiments`, `replica_num` is the number of instances in `2. Launch Inference Instances `

### Results

We list some of the primary evaluation results.  All experiments are conducted on a distributed LLM serving cluster. Each node in the cluster is equipped with 8 Ascend NPUs (910B4: 32 GB HBM or 910B3: 64 GB HBM) and each instance is  equipped with 64GB DRAM for KV cache manager. We evaluate Qwen2.5 7B and 14B models using default float16 precision. Each instance is assigned a dedicated NPU (910B4 for 7B, 910B3 for 14B). DualMap boosts effective request capacity by up to 2.25$\times$ under the same TTFT SLO constraints, compared with the state-of-the-art work.



**Effective request capacity and goodput of different scheduling strategies**

![image-20260129171533430](README.assets/image-20260129171533430.png)

## Citation

If you use DualMap in your research, please cite our paper:

```bibtex
@inproceedings{yuan2026dualmap,
  title={DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving},
  author={Yuan, Ying and Zuo, Pengfei and Wang, Bo and Chen, Zhangyu and Tan, Zhipeng and Yu, Zhou},
  booktitle={Proceedings of the 14th International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
