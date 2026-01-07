SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../benchmarks

python bench_eagle3.py \
    --model meta-llama/Llama-3.3-70B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge \
    --port 30001 \
    --tp 4 \
    --config-list 16,0,0,0 16,3,1,4 \
    --benchmark-list mtbench math500 gsm8k humaneval livecodebench gpqa financeqa \
    --dtype bfloat16 \
    --name llama3-70b-ealge-data-regen

python bench_eagle3.py \
    --model meta-llama/Llama-3.3-70B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path frankleeeee/SGLang-EAGLE3-Llama-3.3-70B-Instruct-perfect-blend \
    --port 30001 \
    --tp 4 \
    --config-list 16,0,0,0 16,3,1,4 \
    --benchmark-list mtbench math500 gsm8k humaneval livecodebench gpqa financeqa \
    --dtype bfloat16 \
    --name llama3-70b-ealge-no-data-regen
