import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import argparse


ACM_FIGSIZE = (1.7, 1.2) 
ACM_FONT = {'family': 'DejaVu Sans', 'size': 8}  
LABEL_SIZE = 8
TICK_SIZE = 8
LINE_WIDTH = 0.5  
BAR_WIDTH = 0.7  
DPI = 300  

ACM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#C47070', '#d62728',  '#999999','#BEB8DC']
ACM_LINESTYLES = ['-', '-', '-', '-','-','-','-']  
ACM_MARKER = ['o', 's', '^', 'd', 'v']  

def calculate_tick_interval(xmax):
    rough_interval = xmax / 5 
    for candidate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 
                      10000, 20000, 50000,
                      100000, 200000, 500000,
                      1000000, 2000000, 5000000,
                      10000000, 20000000, 50000000,
                      100000000, 200000000, 500000000]:
        if rough_interval <= candidate: 
            return candidate
    return 100  

#################### e2e #################################
def clean_df_csv(df, req_start, req_end):
    num_drop = 0
    for x in df.index:
        if (not (req_start <= df.loc[x, "request_id"] <= req_end)) or df.loc[x, "request_latency"] >= 3600000: 
            df.drop(x, inplace=True)
            num_drop += 1
    return df



def save_json_data(path, new_data):
    """Save or merge JSON data to `path`. Creates parent dirs as needed.

    If file exists, load it as dict and update with new_data; otherwise create file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing.update(new_data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(existing, f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)

    return path


def plot_slo_from_csv_json(
    tmp_result_data_dir,
    data_file_path,
    global_scheduler_config_list,
    metrics_dirs,
    metrics_value,
    ttft_slo,
    req_start_id,
    req_end_id,
    xlabel=None,
    ylabel=None,
    title_label=None,
    legend_labels=None,
    ymax=None,
    y_scale_ratio=1.02,
    is_precentage=True,
    is_slo=True
):
    """
    Collect SLO data from CSV and save as structured JSON.

    Args:
        tmp_result_data_dir: Root directory of result data
        data_file_path: Output JSON file path
        global_scheduler_config_list: Scheduler config list
        metrics_dirs: Metrics directory list (aligned with metrics_value)
        metrics_value: Metric values list (e.g. QPS)
        ttft_slo: TTFT SLO threshold (ms)
        req_start_id: Start request ID
        req_end_id: End request ID
    """
    data_label = title_label
    if len(metrics_dirs) != len(metrics_value):
        raise ValueError("metrics_dirs and metrics_value must have the same length")
    data_dict = {}

    if data_label not in data_dict:
        # Save plotting parameters into metadata so plot_from_json can read them later
        data_dict[data_label] = {
            "metadata": {
                "metrics_dirs": metrics_dirs,
                "ttft_slo": ttft_slo,
                "req_range": [req_start_id, req_end_id],
                # plotting hints (may be None)
                "xlabel": xlabel,
                "ylabel": ylabel,
                "title_label": title_label,
                "legend_labels": legend_labels,
                "ymax": ymax,
                "y_scale_ratio": y_scale_ratio,
                "is_precentage": is_precentage,
                "is_slo": is_slo,
            },
            "metrics": sorted(metrics_value),
            "results": {}
        }

    for config in global_scheduler_config_list:
        if config not in data_dict[data_label]["results"]:
            data_dict[data_label]["results"][config] = {}

    for dir_name, metric_val in zip(metrics_dirs, metrics_value):
        for config in global_scheduler_config_list:
            csv_path = os.path.join(tmp_result_data_dir, dir_name, config, "request_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # print(csv_path)
                    df = clean_df_csv(df, req_start_id, req_end_id)
                    
                    condition = (df["time_to_first_token"] <= ttft_slo)
                    proportion = condition.mean() if len(df) > 0 else 0
                    
                    data_dict[data_label]["results"][config][metric_val] = round(proportion,4)
                    print(f"{dir_name} | {config}: SLO attainment {proportion:.2%}")
                except Exception as e:
                    print(f"Error processing {csv_path}: {str(e)}")
                    continue

    save_json_data(data_file_path, data_dict)


def plot_from_json(data_file_path, save_pics_dir, title_label, top_label, x_label_enable, 
                   slo_abalation_qps_list=None, effective_req_cnt_increased=None,qps_increased=None):
    """Read data from JSON file and create a plot.

    This function will prefer plotting parameters stored in the JSON metadata under the
    selected title_label. The following metadata keys are recognized and will override
    function arguments when present: xlabel, ylabel, legend_labels, title_label,
    ymax, y_scale_ratio, is_precentage, is_slo.
    """
    try:
        with open(data_file_path, 'r') as f:
            all_data = json.load(f)
            print(f"all_data: {all_data}")
            print(f"title_label: {data_file_path}")

        if title_label not in all_data:
            raise ValueError(f"Data label '{title_label}' not found in JSON file")

        data = all_data[title_label]
        metadata = data.get('metadata', {})

        # allow metadata to override provided args
        xlabel = metadata.get('xlabel')
        ylabel = metadata.get('ylabel')
        legend_labels = metadata.get('legend_labels')
        title_label = metadata.get('title_label')
        y_scale_ratio = metadata.get('y_scale_ratio')
        is_precentage = metadata.get('is_precentage')
        is_slo = metadata.get('is_slo')
        ymax_hint = metadata.get('ymax')

        # Prepare plot data: {config: {'x': [], 'y': []}}
        plot_data = {}
        for config in data['results']:
            x_vals = []
            y_vals = []
            for metric_val in data['metrics']:
                key = str(metric_val) if str(metric_val) in data['results'][config] else metric_val
                if key in data['results'][config] and data['results'][config][key]:
                    x_vals.append(float(metric_val))
                    y_vals.append(float(data['results'][config][key]))
            if x_vals:
                plot_data[config] = {'x': x_vals, 'y': y_vals}

        if not plot_data:
            raise ValueError('No valid data found for the specified data label')

        ACM_FIGSIZE = (1.7, 1.2)
        ACM_FONT = {'family': 'DejaVu Sans', 'size': 8}
        LABEL_SIZE = 12
        TICK_SIZE = 12
        LINE_WIDTH = 0.5
        BAR_WIDTH = 0.7
        DPI = 300
        ACM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#C47070', '#d62728',  '#999999','#BEB8DC']
        ACM_LINESTYLES = ['-', '-', '-', '-','-','-','-']
        ACM_MARKER = ['o', 's', '^', 'd', 'v']

        ACM_FIGSIZE = (1.7, 1.2)
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': ACM_FONT['size'],
            'axes.titlesize': LABEL_SIZE,
            'axes.labelsize': LABEL_SIZE,
            'xtick.labelsize': TICK_SIZE,
            'ytick.labelsize': TICK_SIZE,
            'legend.fontsize': TICK_SIZE,
            'axes.linewidth': LINE_WIDTH,
        })

        ymax = 0
        ymin = sys.maxsize
        xmax = 0
        xmin = sys.maxsize

        fig = plt.figure(figsize=ACM_FIGSIZE)
        ax = fig.add_axes([0, 0.22, 0.9, 0.9])

        for idx, (config, data_points) in enumerate(plot_data.items()):
            x_vals = data_points['x']
            y_vals = data_points['y']
            plt.plot(x_vals, y_vals,
                     color=ACM_COLORS[idx % len(ACM_COLORS)],
                     linestyle=ACM_LINESTYLES[idx % len(ACM_LINESTYLES)],
                     marker=ACM_MARKER[idx % len(ACM_MARKER)],
                     markersize=5,
                    #  markerfacecolor='none',
                     linewidth=1,
                     label=legend_labels[idx])

            xmax = max(xmax, max(x_vals))
            ymax = max(ymax, max(y_vals))
            ymin = min(ymin, min(y_vals))
            xmin = min(xmin, min(x_vals))

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            try:
                # create a small figure specifically for the legend
                legend_fig = plt.figure(figsize=(3.5, 0.1))
                ax_legend = legend_fig.add_axes([0, 0, 1, 1])
                ax_legend.axis('off')

                ax_legend.legend(handles, labels,
                                loc='center',
                                ncol=len(labels),
                                frameon=False,
                                fontsize=TICK_SIZE - 1,
                                handlelength=1.5,
                                handletextpad=0.5,
                                columnspacing=1.0)
                # ensure output dir exists
                if not os.path.exists(save_pics_dir):
                    os.makedirs(save_pics_dir)
                # save legend next to the main plot and include the title_label to avoid collisions
                safe_title = title_label if title_label else "legend"
                legend_path = os.path.join(save_pics_dir, f"slo_qps_legend.pdf")
                legend_fig.savefig(legend_path, bbox_inches='tight', dpi=DPI, pad_inches=0.02)
                plt.close(legend_fig)
            except Exception as e:
                print(f"Failed to save separate legend: {e}")
        
        def format_y_label(y, max_val, is_precentage):
            """Format Y tick label by value range (percentage, decimal, integer, or scientific)."""
            if is_precentage:
                return f"{int(y*100)}"
            elif max_val < 3:
                return f"{y:.1f}"
            elif 3 <= max_val <= 1000:
                return f"{int(y)}"
            else:
                exponent = int(np.log10(max_val))
                return str(int(y / (10**exponent)))

        if ymax_hint is not None:
            try:
                ymax = float(ymax_hint)
            except Exception:
                pass

        ylim_max = ymax * y_scale_ratio
        ylim_min = 0
        ylim_min = ymin * 0.95
        if is_slo:
            ylim_max = 1

        if ymin == 0:
            ylim_min = 0 - ymax * 0.1
        
        ylim_min = 0

        ax.grid(False)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=LINE_WIDTH)
        ytick_interval = calculate_tick_interval(ylim_max*1.1-ylim_min)
        y_ticks = np.arange(ylim_min, ylim_max*1.1, ytick_interval)
        y_tick_labels = [format_y_label(y, ylim_max, is_precentage) for y in y_ticks]

        ax.set_ylim(ylim_min, ylim_max*1.1)
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Compute X axis ticks
        # Prefer the explicit metrics list from JSON, but generate regular ticks
        # using the same interval logic as Y-axis so labels are 'nice' numbers.
        try:
            metrics_list = [float(v) for v in data.get('metrics', [])]
            if metrics_list:
                xmin = min(metrics_list)
                xmax = max(metrics_list)
        except Exception:
            metrics_list = []

        # compute x tick interval using the same utility
        x_tick_interval = calculate_tick_interval((xmax * 1.1) - xmin)
        x_ticks = np.arange(xmin, xmax * 1.1, x_tick_interval)

        # human-friendly labels: integers without decimal point
        x_tick_labels = [str(int(x)) if float(x).is_integer() else str(x) for x in x_ticks]

        if len(x_ticks) > 0:
            xmin_val = min(x_ticks)
            xmax_val = max(x_ticks)
            ax.set_xlim(xmin_val * 0.8, max(xmax_val * 1.05,xmax * 1.05))
        else:
            ax.set_xlim(xmin * 0.8, xmax * 1.05)

        ax.set_xticks([])
        ax.set_xticklabels([])

        for x, label in zip(x_ticks, x_tick_labels):
            ax.text(x, -0.05,
                    label,
                    ha='center',
                    va='top',
                    fontsize=TICK_SIZE,
                    transform=ax.get_xaxis_transform())

            ax.plot(
                [x, x],
                [-0.03, 0],
                color='black',
                linewidth=LINE_WIDTH,
                clip_on=False,
                transform=ax.get_xaxis_transform(),
                solid_capstyle='butt'
            )

        grid_lines = ax.yaxis.get_gridlines()
        if len(grid_lines) == 0:
            for y in y_ticks:
                ax.axhline(y=y, color='gray', linestyle=':', alpha=0.5, linewidth=LINE_WIDTH)

        ax.set_axisbelow(True)
        fig.canvas.draw()

        for y, label in zip(y_ticks, y_tick_labels):
            ax.text(-0.05, y,
                    label,
                    ha='right',
                    va='center',
                    fontsize=TICK_SIZE,
                    transform=ax.get_yaxis_transform())
            ax.plot(
                [-0.03, 0],
                [y, y],
                color='black',
                linewidth=LINE_WIDTH,
                clip_on=False,
                transform=ax.get_yaxis_transform(),
                solid_capstyle='butt'
            )

        y_exp = int(np.log10(ylim_max)) if ylim_max > 1000 else 0
        if ylim_max > 1000:
            ax.text(-0.15, ylim_max * 1.05, f'$\\times10^{{{y_exp}}}$',
                    fontsize=TICK_SIZE-1,
                    transform=ax.get_yaxis_transform())

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        t = ax.text(0, 0, ylabel, rotation=90, fontsize=LABEL_SIZE, ha='center', va='bottom')
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        text_height_axes = bbox.height / fig.get_figheight() / fig.dpi
        y_pos = 0.5 - text_height_axes / 2
        ymin = 0
        ax.text(-0.3,
                y_pos - 0.15,
                ylabel,
                rotation=90,
                ha='center',
                va='bottom',
                fontsize=LABEL_SIZE,
                transform=ax.transAxes )
        if is_slo:
            ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, zorder=0) # 

        plt.grid(True,color='gray', linestyle=':', alpha=0.5, linewidth=LINE_WIDTH)

        if is_slo and slo_abalation_qps_list:
            for idx, slo_abalation_qps in enumerate(slo_abalation_qps_list):
                try:
                    x_val = float(slo_abalation_qps)
                except Exception:
                    # skip non-numeric entries
                    continue
                # only draw the line if it lies within the plotted x-range (with a small margin)
                try:
                    xlim = ax.get_xlim()
                    margin = (xlim[1] - xlim[0]) * 0.05
                    if x_val < xlim[0] - margin or x_val > xlim[1] + margin:
                        continue
                except Exception:
                    pass

                ax.axvline(x=x_val, ymin=0.0, ymax=0.8,
                           color=ACM_COLORS[idx % len(ACM_COLORS)],
                           linestyle=':', linewidth=1.0, zorder=0)
        
        if top_label:
            ax.set_title(top_label,
                        fontsize=LABEL_SIZE-3,
                        x=0.5,
                        y=0.98)
        if x_label_enable:
            ax.xaxis.set_label_coords(0.5, -0.15)
            plt.xlabel(xlabel)
        if is_slo and effective_req_cnt_increased:
            qps_x = effective_req_cnt_increased[0]
            y_start= effective_req_cnt_increased[1]
            y_end = effective_req_cnt_increased[2]
            v_x= effective_req_cnt_increased[3]
            v_y= effective_req_cnt_increased[4]
            value= effective_req_cnt_increased[5]
            
            ax.annotate('',
                        xy=(qps_x, y_end), xycoords='data',
                        xytext=(qps_x, y_start), textcoords='data',
                        arrowprops=dict(arrowstyle='<->',
                                        color='black',
                                        linewidth=LINE_WIDTH,
                                        shrinkA=0,
                                        shrinkB=0))
            ax.text(
                v_x, v_y, f'+{value}%',
                ha='center', va='bottom',
                fontsize=LABEL_SIZE-1,
                color='black',
                rotation=90,
                rotation_mode='anchor'
            )

        if is_slo and qps_increased:
            qps_xstart = qps_increased[0]
            qps_end= qps_increased[1]
            slo_y = qps_increased[2]
            v_x= qps_increased[3]
            v_y= qps_increased[4]
            value= qps_increased[5]
            
            ax.annotate('',
                        xy=(qps_xstart, slo_y), xycoords='data',
                        xytext=(qps_end, slo_y), textcoords='data',
                        arrowprops=dict(arrowstyle='<->',
                                        color='black',
                                        linewidth=LINE_WIDTH,
                                        shrinkA=0,
                                        shrinkB=0))
            
            ax.text(v_x, v_y, f'+{value}%',
                    ha='left', va='center',
                    fontsize=LABEL_SIZE-1,
                    color='black')


        # plt.tight_layout()
        # Save the plot
        if not os.path.exists(save_pics_dir):
            os.makedirs(save_pics_dir)
        plot_path = os.path.join(save_pics_dir, f"{title_label}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"plot_path: {plot_path}")
        plt.close()

        return plot_path
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        raise



def parse_args():
    parser = argparse.ArgumentParser(description="DualMap experiment launcher")
    parser.add_argument('--model_name', type=str, default="Qwen2.5-7B-Instruct", help='Model name')
    parser.add_argument('--replica_dram_size', type=int, default=64, help='Replica DRAM size in GB')
    parser.add_argument('--dataset_type', type=str, default="toolagent", help='dataset type')
    parser.add_argument('--global_scheduler_type_list', type=str, default="cache_affinity,min_pending_input,min_ttft,preble,dualmap", help='Global scheduler type list, comma separated')
    parser.add_argument('--qps_list', type=str, default="1,2,3,4,5", help='QPS list, comma separated')
    parser.add_argument('--replica_num', type=int, default=8, help='Number of replicas')
    parser.add_argument('--request_num', type=int, default=8000, help='Number of requests to send')
    parser.add_argument('--ttft_slo', type=int, default=5, help='ttft_slo seconds')

    return parser.parse_args()

def main():    

    args = parse_args()
    qps_list = [float(x) for x in args.qps_list.split(",")]
    global_scheduler_config_list = args.global_scheduler_type_list.split(",")
    model_name = args.model_name
    dataset_type = args.dataset_type
    replica_num = args.replica_num
    replica_dram = args.replica_dram_size
    request_num = args.request_num
    ttft_slo = args.ttft_slo

    data_label = model_name
    warm_up_requests_num = 500 
    workspace_path = "./evaluation"
    result_data_path = os.path.join(
        workspace_path,
        'result',
        'data',
        model_name,
        dataset_type,
        f'req{request_num}-warm{warm_up_requests_num}-cache{replica_dram}G'
    )
    processed_result_data = os.path.join(
        workspace_path,
        'result',
        'processed_data',
        model_name,
        dataset_type,
        f'req{request_num}-warm{warm_up_requests_num}-cache{replica_dram}G'
    )


    print(f"global_scheduler_config_list: {result_data_path}")
    print(f"processed_result_data: {processed_result_data}")

    legend_labels = [x.upper() for x in global_scheduler_config_list]

    metrics_dirs = [f"replica{replica_num}-qps{value:.1f}" for value in qps_list]  
    req_start = warm_up_requests_num
    req_end = request_num
    slo_abalation_qps_list = None
    top_label = f"{dataset_type} & {model_name}"
    x_label_enbale = False
    effective_req_cnt_increased = None
    qps_increased = None

    tmp_result_data_dir = result_data_path
    save_pics_dir = os.path.join(processed_result_data, f"pics")
    save_data_dir = os.path.join(processed_result_data, "parsed_data")
    os.makedirs(save_data_dir, exist_ok=True)
    res_json = os.path.join(save_data_dir, f"{data_label}_e2e_qps_res.txt")
    print(f"res_json: {res_json}")

############ e2e ####################
    data_file_path = res_json
    xlabel = "Req rate (req/s)"
    ylabel = "SLO attainment (%)"
    title_label = f"slo_attainment_ttft_{ttft_slo:.1f}"
    plot_slo_from_csv_json(
        tmp_result_data_dir, 
        data_file_path, 
        global_scheduler_config_list,
        metrics_dirs,
        qps_list,
        ttft_slo,
        req_start,
        req_end,
        xlabel=xlabel, ylabel=ylabel,title_label=title_label,legend_labels=legend_labels, 
        y_scale_ratio = 1.02, is_precentage=True, is_slo = True       
    )

    title_label_list = [f"slo_attainment_ttft_{ttft_slo:.1f}"]
    print(f"title_label_list: {title_label_list}")

    for title_label in title_label_list:
        plot_from_json(data_file_path, save_pics_dir, title_label, top_label, x_label_enbale, slo_abalation_qps_list,effective_req_cnt_increased,qps_increased)

if __name__ == "__main__":
    main()