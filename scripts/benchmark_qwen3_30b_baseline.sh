SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

cd $ROOT_DIR/benchmarks
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python3 bench_eagle3.py \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
    --port 50000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list 8,0,0,0 8,3,1,4 8,5,1,6 \
    --benchmark-list mtbench gsm8k math500 humaneval livecodebench financeqa gpqa \
    --dtype bfloat16 \
    --name qwen3_30b
