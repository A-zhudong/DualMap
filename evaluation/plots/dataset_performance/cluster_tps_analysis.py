import argparse
import json
from pathlib import Path

import pandas as pd


INVALID_LATENCY_SENTINEL = 3600000


def load_and_clean(csv_path: Path, req_start=None, req_end=None):
    df = pd.read_csv(csv_path)
    if req_start is not None:
        df = df[df["request_id"] >= req_start]
    if req_end is not None:
        df = df[df["request_id"] <= req_end]

    # Keep only valid finished rows.
    df = df[df["request_latency"] < INVALID_LATENCY_SENTINEL]
    return df.copy()


def compute_cluster_metrics(df: pd.DataFrame, ttft_slo=None):
    if df.empty:
        return {
            "num_requests": 0,
            "makespan_seconds": 0.0,
            "cluster_tps": 0.0,
            "mean_request_tps": 0.0,
            "p50_request_tps": 0.0,
            "p95_request_tps": 0.0,
            "ttft_slo_attainment": None,
            "goodput_req_per_s": None,
        }

    start_ts = float(df["request_start_time"].min())
    end_ts = float(df["request_end_time"].max())
    makespan = max(1e-9, end_ts - start_ts)

    # NOTE: output_len is request target length in current schema.
    total_output_tokens = float(df["output_len"].sum())
    cluster_tps = total_output_tokens / makespan

    req_tps = df["TPS(tokens/s)"]
    out = {
        "num_requests": int(len(df)),
        "makespan_seconds": round(makespan, 6),
        "cluster_tps": round(cluster_tps, 6),
        "mean_request_tps": round(float(req_tps.mean()), 6),
        "p50_request_tps": round(float(req_tps.quantile(0.5)), 6),
        "p95_request_tps": round(float(req_tps.quantile(0.95)), 6),
        "ttft_slo_attainment": None,
        "goodput_req_per_s": None,
    }

    if ttft_slo is not None:
        attainment = float((df["time_to_first_token"] <= ttft_slo).mean())
        goodput = attainment * len(df) / makespan
        out["ttft_slo_attainment"] = round(attainment, 6)
        out["goodput_req_per_s"] = round(goodput, 6)

    return out


def compute_per_replica_metrics(df: pd.DataFrame):
    if df.empty:
        return {}

    out = {}
    for rid, sub in df.groupby("replica_id"):
        out[str(int(rid))] = compute_cluster_metrics(sub)
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze cluster-level TPS from request_metrics.csv")
    parser.add_argument("--csv", required=True, help="Path to request_metrics.csv")
    parser.add_argument("--req_start", type=int, default=None, help="Optional min request_id")
    parser.add_argument("--req_end", type=int, default=None, help="Optional max request_id")
    parser.add_argument("--ttft_slo", type=float, default=None, help="Optional TTFT SLO in seconds")
    parser.add_argument("--include_per_replica", action="store_true", help="Also output per-replica TPS stats")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = load_and_clean(csv_path, args.req_start, args.req_end)

    result = {
        "input_csv": str(csv_path),
        "filters": {
            "req_start": args.req_start,
            "req_end": args.req_end,
            "ttft_slo": args.ttft_slo,
            "dropped_invalid_latency_ge": INVALID_LATENCY_SENTINEL,
        },
        "cluster": compute_cluster_metrics(df, ttft_slo=args.ttft_slo),
    }

    if args.include_per_replica:
        result["per_replica"] = compute_per_replica_metrics(df)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
