SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
# train eagle3 for kimi-2.5
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path moonshotai/Kimi-K2.5 \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/kimi-k2.5-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 8 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template kimi-k2.5 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --log-interval 10 \
    --sglang-mem-fraction-static 0.8
