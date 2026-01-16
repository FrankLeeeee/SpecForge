cd benchmarks

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path frankleeeee/MoE-2experts-topk1-hetero-hetero-0.5-perfectblend_train \
    --port 30001 \
    --config-list 8,3,1,4 8,5,3,6 \
    --benchmark-list mtbench financeqa gpqa math500 gsm8k humaneval livecodebench \
    --dtype bfloat16 \
    --name "moe_2_experts_hetero_0.5"

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path frankleeeee/MoE-2experts-topk1-hetero-hetero-0.7-perfectblend_train \
    --port 30001 \
    --config-list 8,3,1,4 8,5,3,6 \
    --benchmark-list mtbench financeqa gpqa math500 gsm8k humaneval livecodebench \
    --dtype bfloat16 \
    --name "moe_2_experts_hetero_0.7"

python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path frankleeeee/MoE-2experts-topk1-hetero-hetero-0.9-perfectblend_train \
    --port 30001 \
    --config-list 8,3,1,4 8,5,3,6 \
    --benchmark-list mtbench financeqa gpqa math500 gsm8k humaneval livecodebench \
    --dtype bfloat16 \
    --name "moe_2_experts_hetero_0.9"