# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import get_analyzer, find_files_prefix, find_files_suffix
from arguments import parser
import datetime
import numpy as np

# Add performance-specific arguments
parser.add_argument("--tflops_event_key", required=False, default="performance/tflops", type=str, 
                    help="TensorBoard event key for TFLOPs")
parser.add_argument("--memory_event_key", required=False, default="performance/peak_memory_gb", type=str, 
                    help="TensorBoard event key for peak memory consumption")
parser.add_argument("--plot_tflops_name", required=False, default="perf_tflops.png", type=str, 
                    help="Filename for TFLOPs plot")
parser.add_argument("--plot_memory_name", required=False, default="perf_memory.png", type=str, 
                    help="Filename for memory plot")
parser.add_argument("--plot_tflops_pct_name", required=False, default="perf_tflops_pct.png", type=str, 
                    help="Filename for TFLOPs percentage plot")

args = parser.parse_args()

if args.use_sns:
    import seaborn as sns
    sns.set()

def extract_params_from_path(path):
    """Extract training parameters from tensorboard path.
    
    Args:
        path (str): Tensorboard path containing parameter information
        
    Returns:
        dict: Dictionary containing zero_stage, tp, pp, dp, sp, and mbsz values
    """
    # Extract zero stage
    zero_stage = '1'  # default
    for z in ['z1', 'z2', 'z3']:
        if z in path:
            zero_stage = z[1]
            break
            
    # Extract parallelism settings
    tp = path.split('tp')[1].split('_')[0]
    pp = path.split('pp')[1].split('_')[0]
    dp = path.split('dp')[1].split('_')[0]
    sp = path.split('sp')[1].split('_')[0]
    mbsz = path.split('mbsz')[1].split('_')[0]
    
    return {
        'zero_stage': zero_stage,
        'tp': tp,
        'pp': pp,
        'dp': dp,
        'sp': sp,
        'mbsz': mbsz
    }

def get_plot_name(base_name, params):
    """Generate plot name with parameters.
    
    Args:
        base_name (str): Original plot name
        params (dict): Dictionary of parameters
        
    Returns:
        str: Plot name with parameters
    """
    base, ext = os.path.splitext(base_name)
    return f"{base}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}{ext}"

def main():
    # Create dated output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    target_prefix = 'events.out.tfevents'
    tb_log_paths = find_files_prefix(args.tb_dir, target_prefix)
    print(f"Found {len(tb_log_paths)} matching files")
    print(tb_log_paths)
    analyzer = get_analyzer(args.analyzer)
    
    # Store data for each path to calculate percentages later
    all_tflops_data = {}
    max_tflops = 0
    
    # First pass: collect TFLOPs data and find maximum
    for tb_path in tb_log_paths:
        print(f"Processing TFLOPs for: {tb_path}")
        try:
            analyzer.set_names(tb_path)
            event_accumulator = EventAccumulator(tb_path)
            event_accumulator.Reload()
            
            # Process TFLOPs data
            if args.tflops_event_key in event_accumulator.Tags()['scalars']:
                tflops_events = event_accumulator.Scalars(args.tflops_event_key)
                x = [x.step for x in tflops_events]
                y = [x.value for x in tflops_events]
                
                params = extract_params_from_path(tb_path)
                all_tflops_data[tb_path] = {
                    'x': x,
                    'y': y,
                    'params': params,
                    'label': f"{analyzer.get_label_name()}, MBSZ={params['mbsz']}"
                }
                
                # Update max TFLOPs
                path_max_tflops = max(y) if y else 0
                max_tflops = max(max_tflops, path_max_tflops)
                print(f"Max TFLOPs for {tb_path}: {path_max_tflops}")
        except Exception as e:
            print(f"Error processing TFLOPs for {tb_path}: {str(e)}")
    
    print(f"Overall max TFLOPs: {max_tflops}")
    
    # Create TFLOPs plot
    plt.figure(figsize=(12, 8))
    for tb_path, data in all_tflops_data.items():
        plt.plot(data['x'], data['y'], label=data['label'])
        
        if not args.skip_csv:
            df = pd.DataFrame({"step": data['x'], "tflops": data['y']})
            params = data['params']
            csv_filename = os.path.join(output_dir, 
                f"tflops_{args.csv_name}{analyzer.get_csv_filename()}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}.csv")
            df.to_csv(csv_filename)
    
    plt.grid(True)
    if not args.skip_plot:
        plt.legend()
        plt.title(f"{args.plot_title} - TFLOPs")
        plt.xlabel(args.plot_x_label)
        plt.ylabel("TFLOPs")
        if tb_path in all_tflops_data:
            params = all_tflops_data[tb_path]['params']
            plot_name = get_plot_name(args.plot_tflops_name, params)
            plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    # Create TFLOPs percentage plot
    plt.figure(figsize=(12, 8))
    for tb_path, data in all_tflops_data.items():
        # Calculate percentage of max TFLOPs
        y_pct = [100 * val / max_tflops for val in data['y']] if max_tflops > 0 else [0] * len(data['y'])
        plt.plot(data['x'], y_pct, label=data['label'])
        
        if not args.skip_csv:
            df = pd.DataFrame({"step": data['x'], "tflops_pct": y_pct})
            params = data['params']
            csv_filename = os.path.join(output_dir, 
                f"tflops_pct_{args.csv_name}{analyzer.get_csv_filename()}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}.csv")
            df.to_csv(csv_filename)
    
    plt.grid(True)
    if not args.skip_plot:
        plt.legend()
        plt.title(f"{args.plot_title} - TFLOPs (% of Peak)")
        plt.xlabel(args.plot_x_label)
        plt.ylabel("TFLOPs (% of Peak)")
        plt.ylim(0, 105)  # Set y-axis limit to 0-105%
        if tb_path in all_tflops_data:
            params = all_tflops_data[tb_path]['params']
            plot_name = get_plot_name(args.plot_tflops_pct_name, params)
            plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    # Process memory data
    plt.figure(figsize=(12, 8))
    for tb_path in tb_log_paths:
        print(f"Processing memory for: {tb_path}")
        try:
            analyzer.set_names(tb_path)
            event_accumulator = EventAccumulator(tb_path)
            event_accumulator.Reload()
            
            # Process memory data
            if args.memory_event_key in event_accumulator.Tags()['scalars']:
                memory_events = event_accumulator.Scalars(args.memory_event_key)
                x = [x.step for x in memory_events]
                y = [x.value for x in memory_events]
                
                params = extract_params_from_path(tb_path)
                label = f"{analyzer.get_label_name()}, MBSZ={params['mbsz']}"
                
                plt.plot(x, y, label=label)
                
                if not args.skip_csv:
                    df = pd.DataFrame({"step": x, "memory_gb": y})
                    csv_filename = os.path.join(output_dir, 
                        f"memory_{args.csv_name}{analyzer.get_csv_filename()}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}.csv")
                    df.to_csv(csv_filename)
        except Exception as e:
            print(f"Error processing memory for {tb_path}: {str(e)}")
    
    plt.grid(True)
    if not args.skip_plot:
        plt.legend()
        plt.title(f"{args.plot_title} - Peak Memory")
        plt.xlabel(args.plot_x_label)
        plt.ylabel("Peak Memory (GB)")
        if 'params' in locals():
            plot_name = get_plot_name(args.plot_memory_name, params)
            plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()

def plot_csv():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    target_suffix = 'csv'
    csv_log_files = find_files_suffix(args.csv_dir, target_suffix)
    
    analyzer = get_analyzer(args.analyzer)
    
    # Process TFLOPs data
    plt.figure(figsize=(12, 8))
    tflops_files = [f for f in csv_log_files if f.startswith('tflops_') and not f.startswith('tflops_pct_')]
    max_tflops = 0
    tflops_data = {}
    
    # First pass to find max TFLOPs
    for csv_file in tflops_files:
        analyzer.set_names(csv_file)
        x, y = [], []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == 'step':
                    continue
                x.append(int(row[1]))
                y.append(float(row[2]))
        
        tflops_data[csv_file] = {'x': x, 'y': y}
        file_max_tflops = max(y) if y else 0
        max_tflops = max(max_tflops, file_max_tflops)
    
    # Plot TFLOPs
    for csv_file, data in tflops_data.items():
        plt.plot(data['x'], data['y'], label=f'{analyzer.get_label_name()}')
    
    plt.grid(True)
    plt.legend()
    plt.title(f"{args.plot_title} - TFLOPs")
    plt.xlabel(args.plot_x_label)
    plt.ylabel("TFLOPs")
    plt.savefig(os.path.join(output_dir, args.plot_tflops_name))
    plt.close()
    
    # Plot TFLOPs percentage
    plt.figure(figsize=(12, 8))
    for csv_file, data in tflops_data.items():
        y_pct = [100 * val / max_tflops for val in data['y']] if max_tflops > 0 else [0] * len(data['y'])
        plt.plot(data['x'], y_pct, label=f'{analyzer.get_label_name()}')
    
    plt.grid(True)
    plt.legend()
    plt.title(f"{args.plot_title} - TFLOPs (% of Peak)")
    plt.xlabel(args.plot_x_label)
    plt.ylabel("TFLOPs (% of Peak)")
    plt.ylim(0, 105)  # Set y-axis limit to 0-105%
    plt.savefig(os.path.join(output_dir, args.plot_tflops_pct_name))
    plt.close()
    
    # Plot memory data
    plt.figure(figsize=(12, 8))
    memory_files = [f for f in csv_log_files if f.startswith('memory_')]
    
    for csv_file in memory_files:
        analyzer.set_names(csv_file)
        x, y = [], []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == 'step':
                    continue
                x.append(int(row[1]))
                y.append(float(row[2]))
        
        plt.plot(x, y, label=f'{analyzer.get_label_name()}')
    
    plt.grid(True)
    plt.legend()
    plt.title(f"{args.plot_title} - Peak Memory")
    plt.xlabel(args.plot_x_label)
    plt.ylabel("Peak Memory (GB)")
    plt.savefig(os.path.join(output_dir, args.plot_memory_name))
    plt.close()

if __name__ == "__main__":
    if args.plot_only:
        plot_csv()
    else:
        main()