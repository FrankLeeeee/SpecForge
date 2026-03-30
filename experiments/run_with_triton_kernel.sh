SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
cd $ROOT_DIR/benchmarks

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SGLANG_ENABLE_SPEC_V2=1

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path ../outputs/llama3_8b_torch_kernel/epoch_1_step_15000 \
    --port 50002 \
    --config-list 8,0,0,0 8,3,1,4 \
    --benchmark-list mtbench gsm8k math500 humaneval livecodebench gpqa financeqa \
    --dtype bfloat16 \
    --name llama3-8b-torch-kernel