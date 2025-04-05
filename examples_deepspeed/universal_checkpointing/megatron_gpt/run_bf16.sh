#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
BASE_DATA_PATH=dataset
DATASET=${BASE_DATA_PATH}/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

# Check if ZERO_STAGE is set, default to 0 if not
ZERO_STAGE=${ZERO_STAGE:-0}

# Validate ZERO_STAGE
if [[ ! $ZERO_STAGE =~ ^[0-3]$ ]]; then
    echo "Error: ZERO_STAGE must be 0, 1, 2, or 3"
    exit 1
fi

DTYPE="bf16"
EXIT_INTERVAL=200

# Debug
DEBUG_MODE=${DEBUG_MODE:-0}
if [[ $DEBUG_MODE == 1 ]]; then
        LAYERS=4
        HIDDEN=512
        SEQ=512
        SIZE_TAG="gpt-117m"
else
        LAYERS=48
        HIDDEN=1024
        SEQ=1024
        SIZE_TAG="gpt-1.5b"
fi  # Add this missing 'fi' to close the if statement

# 3D parallelism of training 
TP=${TP:-1}  
PP=${PP:-1}  
DP=${DP:-1}  
SP=${SP:-1}  
WORLD_SIZE=$((TP*PP*DP*SP))
MICRO_BATCH=${MICRO_BATCH:-4}
GLOBAL_BATCH=$((MICRO_BATCH*WORLD_SIZE))
TRAIN_ITERS=100000
LR=6.0e-3
MIN_LR=6.0e-4

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP
RUN_TAG="save"
# RUN_TAG="ref_load${LOAD_TP}_${LOAD_PP}_${LOAD_DP}"

EXP_DIR="z${ZERO_STAGE}_uni_ckpt" 
CHECKPOINT_PATH=${EXP_DIR}/checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_mb${MICRO_BATCH}_${SIZE_TAG}
LOAD_CHECKPOINT_PATH=${EXP_DIR}/checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${LOAD_TP}_pp${LOAD_PP}_dp${LOAD_DP}_sp${LOAD_SP}_mb${MICRO_BATCH}_${SIZE_TAG}
LOG_DIR="${EXP_DIR}/tensorboard/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_${SIZE_TAG}_${RUN_TAG}"
mkdir -p $LOG_DIR

# Define log file with timestamp and configuration details
MEGATRON_LOG_FILE="${LOG_DIR}/megatron_${DATETIME}_tp${TP}_pp${PP}_dp${DP}_sp${SP}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_${DTYPE}_${SIZE_TAG}.log"

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
	--ds-sequence-parallel-size $SP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads 16 \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters $TRAIN_ITERS \
        --lr $LR \
	--min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --log-memory-to-tensorboard \
        --eval-iters 40 \
        --eval-interval 10 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 100 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --${DTYPE} \
	--checkpoint-activations \
	--exit-interval ${EXIT_INTERVAL} \
        --save ${CHECKPOINT_PATH} \
        --load ${LOAD_CHECKPOINT_PATH} \
        --make-vocab-size-divisible-by 256 \
	--tensorboard-dir $LOG_DIR
        "

options="${options} \
        --deepspeed \
        --deepspeed_config=${CONFIG_JSON} \
        --zero-stage=${ZERO_STAGE} \
        --deepspeed-activation-checkpointing \
"
if [[ ${ZERO_STAGE} -gt 1 ]]; then
options="${options} \
    --no-pipeline-parallel"
fi

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  },

  "data_types": {
        "grad_accum_dtype": "fp32" 
  },

  "wall_clock_breakdown" : false
}
EOT

WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
run_cmd="deepspeed --master_port 29700 $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options} 2>&1 | tee ${MEGATRON_LOG_FILE}"

echo "Logging output to: ${MEGATRON_LOG_FILE}"
echo ${options}
echo ${run_cmd}
eval ${run_cmd}

set +x
