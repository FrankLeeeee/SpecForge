SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

cd $ROOT_DIR/benchmarks

export SGLANG_ENABLE_SPEC_V2=1
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --port 40001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 8,3,1,4 8,5,1,6 8,7,1,8 \
    --benchmark-list mtbench gsm8k math500 humaneval livecodebench financeqa gpqa \
    --dtype bfloat16 \
    --name llama3_8b_baseline
