ROOT_DIR=$PWD

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path /data/shenggui/projects/spec-decoding/EAGLE/eagle/traineagle3/sharegpt_expanded.jsonl \
    --build-dataset-num-proc 64 \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-e2e \
    --num-epochs 2 \
    --batch-size 8 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --log-interval 10
