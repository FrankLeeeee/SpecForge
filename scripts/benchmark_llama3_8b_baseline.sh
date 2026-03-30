SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

cd $ROOT_DIR/benchmarks
export SGLANG_ENABLE_SPEC_V2=1
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge \
    --port 40000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 8,0,0,0 \
    --benchmark-list mtbench gsm8k math500 humaneval livecodebench financeqa gpqa \
    --dtype bfloat16 \
    --name llama3_8b_no_eagle
