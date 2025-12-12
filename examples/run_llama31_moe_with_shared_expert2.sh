SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for llama3.1-8b
NUM_GPUS=4
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
NAME=MoE-2experts-topk1-shared

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /workdir/huggingface.co/meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3-with-shared-moe-2.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/$NAME \
    --num-epochs 3 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --target-model-backend sglang \
    --log-interval 10 \
    --report-to wandb \
    --sglang-mem-fraction-static 0.5 \
    --wandb-project specforge-moe-ablation-studies \
    --wandb-name $NAME
    # Optional: Initialize MOE experts from a dense model checkpoint
    # --dense-model-path /path/to/dense/model/checkpoint
