#!/bin/bash
set -e
declare -a CMD

rocm-smi

MASTER_PORT=29501
export DGXNNODES=$SLURM_NNODES
CMD="torchrun --nproc_per_node=8  --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
echo "COMMAND: $CMD"
echo "Starting with seed $SEED on Rank: $SLURM_NODEID"

env | sort | awk -F= '{ printf "    %s=%s\n", $1, $2 }' | tee /logs/env.txt

# ================ Set up environment ============
if [[ $PROFILER == "rpd" ]]; then
    echo "Profiler is set to $PROFILER, tracer will be created at $(pwd)"
    bash loadTracer.sh
fi

# ================ Start timing ================
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
$CMD src/train.py
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# =============== End timing ================
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# ================= Report Results ================
result=$(( end - start ))
result_name="LLM_FINETUNING"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

# ================ Post Processing [#TODO @sarora to think of pythonic post-hooks] ================
if [[ $PROFILER == "rpd" ]]; then
    echo "Post processing RPD trace"
    if [[ ! -f trace.rpd ]]; then
        echo "RPD trace file not found. Skipping post processing."
        exit 0
    fi
    sqlite3 -header -csv trace.rpd "select * from top;" > trace.csv
    source /workspace/utilities/metrics/profiling/add_trace_csv_header.sh
    add_trace_csv_header "LLAMA2_SFT"
    python3 /workspace/deps/rocmProfileData/tools/rpd2tracing.py --start 49% --end 50% trace.rpd trace_49_50.json
    python3 /workspace/deps/rocmProfileData/tools/rpd2tracing.py --start 89% --end 90% trace.rpd trace_89_90.json
    python3 /workspace/deps/rocmProfileData/tools/rpd2tracing.py --start 10% --end 11% trace.rpd trace_10_11.json
fi
    if [[ -d /logs ]]; then
        echo "Moving trace files to /logs"
        mv trace.{rpd,csv} trace_*.json /logs/ 2>/dev/null || true
    else
        echo "Logs directory not found. Skipping moving trace files."
    fi

exit 0
