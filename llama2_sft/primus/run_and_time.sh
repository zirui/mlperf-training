#!/bin/bash

set -e

mkdir -p /results

cd /workspace/Primus

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)

# Build CLI extra args for pretrained checkpoint
POSTTRAIN_CLI_EXTRA=""
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    POSTTRAIN_CLI_EXTRA="modules.post_trainer.overrides.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
fi

# Launch training via primus-cli
./runner/primus-cli direct train posttrain \
    --config ${EXP} \
    ${POSTTRAIN_CLI_EXTRA}

ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)

# Report result
result=$(( end - start ))
result_name="LLAMA2_70B_LORA_SFT"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
