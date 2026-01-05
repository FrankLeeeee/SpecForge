ROOT_DIR=$PWD
# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-30b-a3b-eagle3-sharegpt-sglang \
    --num-epochs 3 \
    --batch-size 1 \
    --tp-size 4 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --log-interval 10 \
    --sglang-ep-size 4
