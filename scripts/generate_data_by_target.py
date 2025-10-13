"""
Simple script to generate responses using local OpenAI API from JSONL file.

python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-20b \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
    --mem-frac=0.4 \
    --tp 8 \
    --host 0.0.0.0 \
    --port 30001 \
    --attention-backend fa3 \
    --reasoning-parser gpt-oss

python scripts/generate_data_by_target.py \
    --model-name openai/gpt-oss-20b \
    --raw-data-file ./cache/dataset/sharegpt_train.jsonl \
    --output-dir ./cache/regenerated_dataset/sharegpt-gpt-oss-20b \
    --max-concurrency 512 \
    --num-per-shard 20000 \
    --server-address-port 127.0.0.1:30001 \
    --is-reasoning-model \
    --is-gpt-oss
"""

import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI, OpenAIError
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = ""


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--num-per-shard", type=int, default=50_000)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16 * 1024)
    parser.add_argument(
        "--server-address-port", type=str, nargs="+", default=["127.0.0.1:30000"]
    )
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")
    return parser.parse_args()


@dataclass
class RequestFuncObject:
    """ """

    conversation_id: str
    input_conversations: List[Dict[str, str]]
    model_name: str
    system_prompt: Optional[str]
    temperature: float = 0.0
    max_tokens: int = 16 * 1024
    output_conversations: Optional[List[Dict[str, str]]] = None
    output_tokens: int = 0
    error: Optional[str] = None
    is_reasoning_model: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)


async def build_conversation(
    req_obj: RequestFuncObject, client: AsyncOpenAI, pbar: Optional[tqdm] = None
) -> RequestFuncObject:
    """
    Call SGLang to retrieve the response for the input connversations.

    Args:
        req_obj (RequestFuncObject): The request object for the current conversation, used to store the meta data and conversation history.
        client (AsyncOpenAI): The client object to use to call the SGLang API to retrieve the response.
        pbar (Optional[tqdm]): The tqdm progress bar to update.

    Returns:
        RequestFuncObject: The updated request object
    """
    messages = []
    if req_obj.system_prompt is not None and len(req_obj.system_prompt) > 0:
        # add the system prompt if given
        messages.append({"role": "system", "content": req_obj.system_prompt})

    req_obj.output_tokens = 0
    for message in req_obj.input_conversations:
        if message["role"] == "assistant":
            # skip, it will be overriden by the generated response
            continue

        if message["role"] == "user":
            messages.append({"role": "user", "content": message["content"]})
            try:
                response = await client.chat.completions.create(
                    model=req_obj.model_name,
                    messages=messages,
                    max_tokens=req_obj.max_tokens,
                    temperature=req_obj.temperature,
                    stream=False,
                )
                response_text = response.choices[0].message.content
                req_obj.output_tokens += response.usage.completion_tokens

                if req_obj.is_reasoning_model:
                    # used for gpt-oss like reasoning models
                    reasoning_content = response.choices[0].message.reasoning_content
            except Exception as e:
                req_obj.error = str(e)
                break

            # store the response
            msg = {"role": "assistant", "content": response_text}
            if req_obj.is_reasoning_model:
                msg["thinking"] = reasoning_content
            messages.append(msg)

    if pbar is not None:
        pbar.update(1)
    req_obj.output_conversations = messages
    return req_obj


async def limited_build_conversation(
    req_obj: RequestFuncObject,
    client: AsyncOpenAI,
    semaphore: Optional[asyncio.Semaphore] = None,
    pbar: Optional[tqdm] = None,
) -> RequestFuncObject:
    """
    Asynchronously call the build_conversation function with a semaphore if given.

    Args:
        req_obj (RequestFuncObject): The request object for the current conversation, used to store the meta data and conversation history.
        client (AsyncOpenAI): The client object to use to call the SGLang API to retrieve the response.
        semaphore (Optional[asyncio.Semaphore]): The semaphore to use to limit the number of concurrent requests.
        pbar (Optional[tqdm]): The tqdm progress bar to update.

    Returns:
        RequestFuncObject: The updated request object
    """
    if semaphore is None:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)
    async with semaphore:
        return await build_conversation(req_obj=req_obj, client=client, pbar=pbar)


def get_random_temperature(
    temperature_choices: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
    temperature_weights: List[int] = [4, 1, 1, 1, 3],
) -> float:
    """
    Get a random temperature for the model generation.

    Args:
        temperature_choices (List[float]): The list of temperatures to choose from.
        temperature_weights (List[int]): The list of weights for the temperatures.

    Returns:
        float: The random temperature
    """
    return random.choices(temperature_choices, weights=temperature_weights)[0]


def get_random_reasoning_effort(
    reasoning_efforts: List[str] = ["low", "medium", "high"],
    reasoning_weights: List[int] = [3, 6, 1],
) -> str:
    """
    Get a random reasoning effort level for the model with weighted probabilities. This is typically only used for gpt-oss like reasoning models.

    Args:
        reasoning_efforts (List[str]): The list of reasoning efforts to choose from.
        reasoning_weights (List[int]): The list of weights for the reasoning efforts.

    Returns:
        str: The random reasoning effort level
    """
    return random.choices(reasoning_efforts, weights=reasoning_weights)[0]


async def main():
    # initialize the arguments
    args = parse_args()
    if args.is_gpt_oss:
        args.is_reasoning_model = True
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    total_ds = load_dataset("json", data_files=args.raw_data_file)["train"]

    # warmup the server by sending dummy requests
    for start in range(0, len(total_ds), args.num_per_shard):
        # get the current shard of dataset
        end = min(start + args.num_per_shard, len(total_ds))
        output_file = os.path.join(args.output_dir, f"shard_{start}-{end}.jsonl")
        output_file_error = os.path.join(args.output_dir, f"error_{start}-{end}.jsonl")
        if os.path.exists(output_file):
            logger.warning(
                f"Skipping generate data from {start} to {end} as {output_file} already exists"
            )
            continue
        ds = total_ds.select(range(start, end))
        logger.info(f"Generating data from {start} to {end}")

        # create clients and warmup the server
        pbar = None if args.disable_tqdm else tqdm(total=len(ds))
        client_semaphore_list = []
        for server_address_port in args.server_address_port:
            client = AsyncOpenAI(
                base_url=f"http://{server_address_port}/v1", api_key="None"
            )
            semaphore = (
                asyncio.Semaphore(args.max_concurrency)
                if args.max_concurrency
                else None
            )
            # send a dummy request to the server to warm up
            try:
                resp = await client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    max_tokens=args.max_tokens,
                    temperature=get_random_temperature(),
                )
                assert resp.choices[0].message.content is not None
                print(
                    f"Dummy request successful for {server_address_port}: {resp.choices[0].message.content}"
                )
            except Exception as e:
                print(
                    f"Warning: Dummy request failed for {server_address_port}: {e}, will ignore this server"
                )
                continue
            client_semaphore_list.append((client, semaphore))
        assert len(client_semaphore_list) > 0, "No server address port is available"

        # call the llm server asynchronously
        tasks = []
        for i, row in enumerate(ds):
            client, semaphore = client_semaphore_list[i % len(client_semaphore_list)]
            req_obj = RequestFuncObject(
                conversation_id=str(row["id"]),
                input_conversations=row["conversations"],
                model_name=args.model_name,
                system_prompt=SYSTEM_PROMPT,
                temperature=get_random_temperature(),
                max_tokens=args.max_tokens,
                is_reasoning_model=args.is_reasoning_model,
                extra_body=(
                    {"reasoning_effort": get_random_reasoning_effort()}
                    if args.is_gpt_oss
                    else {}
                ),
            )
            tasks.append(
                asyncio.create_task(
                    limited_build_conversation(
                        req_obj=req_obj, client=client, semaphore=semaphore, pbar=pbar
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # save the outputs and errors to output_file and output_file_error respectively
        with open(output_file, "w") as f, open(output_file_error, "a") as f_error:
            for output_obj in outputs:
                output_dict = {
                    "conversation_id": output_obj.conversation_id,
                    "conversations": output_obj.output_conversations,
                }
                if args.is_gpt_oss:
                    output_dict["reasoning_effort"] = output_obj.extra_body[
                        "reasoning_effort"
                    ]
                if output_obj.error is not None:
                    output_dict["error"] = output_obj.error
                    output_dict["input_conversations"] = output_obj.input_conversations
                    f_error.write(json.dumps(output_dict) + "\n")
                else:
                    f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
