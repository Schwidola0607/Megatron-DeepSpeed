#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

OUTPUT_PATH=$1
START_ITERATION=$2

if [ "$OUTPUT_PATH" == "" ]; then
    OUTPUT_PATH="z1_uni_ckpt"
fi

# Training Loss
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis_script.py \
    --tb_dir $OUTPUT_PATH \
    --tb_event_key "lm-loss-training/lm loss" \
    --plot_name "uc_char_training_loss.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Training Loss" \
    --start_iteration $START_ITERATION

# Validation Loss
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis_script.py \
    --tb_dir $OUTPUT_PATH \
    --tb_event_key "lm-loss-validation/lm loss validation" \
    --csv_name "val_" \
    --plot_name "uc_char_validation_loss.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Validation Loss" \
    --plot_y_label "Validation LM Loss" \
    --start_iteration $START_ITERATION
