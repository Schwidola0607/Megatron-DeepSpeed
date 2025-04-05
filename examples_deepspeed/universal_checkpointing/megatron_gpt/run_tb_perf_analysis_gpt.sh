#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

OUTPUT_PATH=$1

if [ "$OUTPUT_PATH" == "" ]; then
    OUTPUT_PATH="z1_uni_ckpt"
fi

# TFLOPs Analysis
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis_perf_script.py \
    --tb_dir $OUTPUT_PATH \
    --tflops_event_key "performance/tflops" \
    --plot_tflops_name "uc_perf_tflops.png" \
    --plot_tflops_pct_name "uc_perf_tflops_pct.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Performance" \
    --plot_x_label "Training Step" \

# Memory Analysis
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis_perf_script.py \
    --tb_dir $OUTPUT_PATH \
    --memory_event_key "performance/peak_memory_gb" \
    --plot_memory_name "uc_perf_memory.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Memory" \
    --plot_y_label "Peak Memory (GB)" \
    --plot_x_label "Training Step" \

# Plot from CSV (if needed)
# python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis_perf_script.py \
#     --plot_only \
#     --csv_dir "/path/to/csv/files" \
#     --plot_title "Megatron-GPT Universal Checkpointing - Performance from CSV" \