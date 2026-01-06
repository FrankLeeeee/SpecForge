ROOT_DIR=$PWD
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

export WANDB_API_KEY=d38075491c84c0774138377d6ff2e94befa16324
WANDB_PROJECT=specforge-precision-impact
WANDB_NAME=llama3-8b-dense-sharegpt-sglang

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-sharegpt-sglang \
    --num-epochs 3 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --log-interval 3 \
    --sglang-mem-fraction-static 0.25 \
    --report-to wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-name $WANDB_NAME
