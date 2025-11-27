SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../..
NUM_GPUS=${1:-8}


export WANDB_API_KEY=d38075491c84c0774138377d6ff2e94befa16324
WANDB_NAME="llama3-8b-constant-ttt"

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/fast-eagle/constant-ttt \
    --num-epochs 2 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --report-to wandb \
    --wandb-project fast-eagle \
    --wandb-name ${WANDB_NAME} \
    --log-interval 50 \
    --profile \
    --dist-timeout 60 \
    --ttt-scheduleh constant
