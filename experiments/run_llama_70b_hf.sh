ROOT_DIR=$PWD

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.3-70B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-70B-ealge3.json \
    --train-data-path /data/shenggui/projects/spec-decoding/EAGLE/eagle/traineagle3/sharegpt_expanded.jsonl \
    --build-dataset-num-proc 64 \
    --output-dir $ROOT_DIR/outputs/llama3-70b-eagle3-e2e \
    --num-epochs 2 \
    --batch-size 2 \
    --tp-size 4 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --target-model-backend hf \
    --log-interval 10
