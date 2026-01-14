ROOT_DIR=$PWD

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path /data/shenggui/projects/spec-decoding/EAGLE/eagle/traineagle3/sharegpt_expanded.jsonl \
    --build-dataset-num-proc 64 \
    --output-dir $ROOT_DIR/outputs/qwen3-30b-a3b-eagle3-e2e \
    --num-epochs 2 \
    --batch-size 4 \
    --tp-size 4 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --log-interval 10
