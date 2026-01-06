# Source - https://stackoverflow.com/a
# Posted by dogbane, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-06, License - CC BY-SA 4.0

#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../benchmarks


python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path ../outputs/llama3-8b-eagle3-sharegpt-sglang/epoch_2_step_45000/ \
    --port 30000 \
    --config-list 8,3,1,4 \
    --benchmark-list mtbench math500 gsm8k humaneval \
    --dtype bfloat16
