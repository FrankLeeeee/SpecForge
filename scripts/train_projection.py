import argparse
import hashlib
import os
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from specforge.tracker import NoOpTracker


from specforge import (
    AutoDraftModelConfig,
)
from specforge.data import (
    build_eagle3_dataset,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    TargetHead,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    print_with_rank,
    rank_0_priority,
)
from torch.nn import functional as F

def parse_args() -> Tuple[ArgumentParser, Namespace]:
    """
    This function is used to parse the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )

    # dataset arguments
    parser.add_argument("--train-data-path", type=str, required=True)

    # training hyper params
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="flex_attention")

    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # swanlab-specific args
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    # vlm related args
    parser.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    parser.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="The backend of the target model",
    )

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    """
    Build the experiment tracker according to the report_to argument.

    Args:
        args: The arguments for the training script.
        parser: The parser for the training script.

    Returns:
        The experiment tracker.
    """
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig
) -> Tuple[Union[Eagle3TargetModel, TargetHead], Optional[AutoProcessor]]:
    """
    Build the target model according to the arguments.

    Args:
        args: The arguments for the training script.
        draft_model_config: The draft model config.

    Returns:
        The target model.
    """
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.cache_dir,
    )

    # set the aux hidden states layers
    target_model_config = AutoConfig.from_pretrained(args.target_model_path)
    num_layers = target_model_config.num_hidden_layers
    aux_hidden_states_layers = [1, num_layers // 2 - 1, num_layers - 4, num_layers - 2]
    target_model.set_aux_hidden_states_layers(aux_hidden_states_layers)
    return target_model


def sanity_check(args: Namespace) -> None:
    """
    Perform sanity checks on the arguments.

    Args:
        args: The arguments for the training script.

    Returns:
        None
    """
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size

def build_dataloaders(
    args: Namespace,
) -> DataLoader:
    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=False,
            is_preformatted=args.is_preformatted,
            processor=None,
            num_proc=args.build_dataset_num_proc,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=False,
    )

    return train_dataloader


def run_forward(
    args: Namespace,
    projection_layer: nn.Module,
    data: dict,
    target_model: Eagle3TargetModel,
) -> torch.Tensor:
    _, _, aux_hidden_states_list, last_hidden_states_list = target_model.extend(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
        return_last_hidden_states=True,
        return_logits=False,
    )
    aux_hidden_states_list = [
        val.unsqueeze(0).chunk(4, dim=-1) for val in aux_hidden_states_list
    ]

    inputs = torch.cat(
        [
            torch.cat(val[:3], dim=-1) for val in aux_hidden_states_list
        ],
        dim=0
    )
    labels = torch.cat(
        [
            val[3] for val in aux_hidden_states_list
        ],
        dim=0
    )
    output = projection_layer(inputs)
    loss = F.mse_loss(output, labels)
    return loss

def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]

def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    sanity_check(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    target_model = build_target_model(args, draft_model_config)
    projection_layer = nn.Linear(3 * draft_model_config.hidden_size, draft_model_config.hidden_size, bias=False).cuda()

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader = build_dataloaders(args)

    projection_layer = DDP(
        projection_layer,
    ).to(torch.bfloat16)
    print_with_rank("Initialized projection layer DDP model")

    total_steps = args.num_epochs * len(train_dataloader)

    # ================================================
    # 5. Build optimizer and scheduler
    # ================================================
    optimizer = BF16Optimizer(
        projection_layer,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    dist.barrier()
    os.makedirs(args.output_dir, exist_ok=True)
    print_with_rank(f"Output directory: {args.output_dir}")

    # ================================================
    # 7. Start training
    # ================================================
    global_step = 0
    for epoch in range(args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        projection_layer.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            global_step += 1

            # ================================================
            # 7.1 Training Step
            # ================================================
            loss = run_forward(
                args, projection_layer, data, target_model
            )
            loss.backward()
            optimizer.step()

            # log training metrics
            if global_step % args.log_interval == 0 and not isinstance(tracker, NoOpTracker):
                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.get_learning_rate(),
                }
                tracker.log(log_dict, step=global_step)

            # ================================================
            # 7.3 Save Checkpoints
            # ================================================
            if global_step % args.save_interval == 0:
                # Save the model
                if dist.get_rank() == 0:
                    ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}_step_{global_step}.pth")
                    torch.save(projection_layer.state_dict(), ckpt_path)
                    print_with_rank(f"Saved projection layer checkpoint to {ckpt_path}")
                dist.barrier()

    # Close the tracker
    tracker.close()
    destroy_distributed()

if __name__ == "__main__":
    main()
