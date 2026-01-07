SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../benchmarks

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge \
    --port 30000 \
    --config-list 8,0,0,0 8,3,1,4 \
    --benchmark-list mtbench math500 gsm8k humaneval livecodebench gpqa financeqa \
    --dtype bfloat16 \
    --name llama3-8b-ealge-data-regen

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path frankleeeee/SGLang-EAGLE3-Llama-3.1-8B-Instruct-perfect-blend \
    --port 30000 \
    --config-list 8,0,0,0 8,3,1,4 \
    --benchmark-list mtbench math500 gsm8k humaneval livecodebench gpqa financeqa \
    --dtype bfloat16 \
    --name llama3-8b-ealge-no-data-regen
