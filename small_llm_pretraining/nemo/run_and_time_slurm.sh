set -e
declare -a DATA_CMD
declare -a TRAIN_CMD

MASTER_PORT=${MASTER_PORT:-29501}
export DGXNNODES=${SLURM_NNODES:-$NNODES}

if [ "$SLURM_NNODES" ] && [ "$SLURM_NODEID" ]; then
    # Running under SLURM
    CMD="torchrun --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
    echo "SLURM detected - Starting with seed $SEED on Rank: $SLURM_NODEID"
else
    # Local multi-GPU or single node
    CMD="torchrun"
    echo "Local execution - Starting with $NNODES nodes, $GPUS_PER_NODE GPUs per node"
fi

DATA_CMD="torchrun --nnodes=1 --nproc_per_node=1"
TRAIN_CMD="$CMD --nnodes=$DGXNNODES --nproc_per_node=$GPUS_PER_NODE"

echo "PREPARE COMMAND: $DATA_CMD"
echo "TRAIN COMMAND: $TRAIN_CMD"

# Log environment variables
env | sort | awk -F= '{ printf "    %s=%s\n", $1, $2 }' | tee /logs/env.txt 2>/dev/null || true

CMD_SUFFIX=""

if [ $USE_CKPT -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --use_ckpt"
    if [ $FROM_HF -gt 0 ]; then
        CMD_SUFFIX="${CMD_SUFFIX} --resume_from_hf"
    fi
fi

if [ $SAVE_CKPT -gt 0 ]; then 
    CMD_SUFFIX="${CMD_SUFFIX} --save_ckpt"
fi

if [ ! $MAX_STEPS = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --max_steps ${MAX_STEPS}"
fi

if [ ! $TAG = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tag ${TAG}"
fi

if [ $TP -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tensor_parallel_size ${TP}"
fi

if [ $PP -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --pipeline_parallel_size ${PP}"
fi

if [ $CP -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --context_parallel_size ${CP}"
fi

if [ $FP8_PARAMS -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --fp8_params"
fi

# Allows MLLogger objects to be constructed locally
if [ ! -d /mlperf-outputs ]; then mkdir /mlperf-outputs; fi

set -x

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

$DATA_CMD src/prepare_data.py \
--gbs $GBS --mbs $MBS \
--seed $SEED \
--eval_every $EVAL_EVERY \
--start_eval_at $START_EVAL_AT
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

$TRAIN_CMD src/train.py \
--nodes $DGXNNODES \
--gpus_per_node $GPUS_PER_NODE \
--size $SIZE \
--gbs $GBS --mbs $MBS \
--max_lr $MAX_LR \
--seed $SEED \
--continual_ckpt_path $CONTINUAL_CKPT \
--target_log_ppl $TARGET \
--step_time_atol $STEP_TIME_ATOL \
--ckpt_start_step $START_STEPS \
--warmup_steps $WARMUP_STEPS \
--eval_every $EVAL_EVERY \
--start_eval_at $START_EVAL_AT \
$CMD_SUFFIX
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( end - start ))
result_name="LLM_FINETUNING"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0
