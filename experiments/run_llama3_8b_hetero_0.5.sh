ROOT_DIR=$PWD

# train eagle3 for llama3.1-8b

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3-moe-hetero-experts-0.5.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_regenerated.jsonl \
    --build-dataset-num-proc 64 \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-moe-hetero-experts-0.5 \
    --num-epochs 2 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --log-interval 10 \
    --report-to wandb \
    --wandb-project specforge-moe-ablation-studies \
    --wandb-name llama3-8b-hetero-0.5 \
